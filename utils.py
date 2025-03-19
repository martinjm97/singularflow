from tap import Tap
from typing import Any
import jax.numpy as jnp
import jax
from jax import random
from jax.example_libraries import optimizers as jax_opt
import matplotlib.pyplot as plt
from typing import Literal
from decimal import Decimal
from jax.tree_util import PyTreeDef
from jax.tree_util import tree_flatten, tree_unflatten


def flatten_nn_params(params) -> tuple[jnp.ndarray, PyTreeDef]:
    flat_params = [(w.tolist(), b.tolist()) for w, b in params]
    flat_params, tree_def = tree_flatten(flat_params)
    flat_params = jnp.array(flat_params)
    return flat_params, tree_def


def naca4(x, args):
    m, p, t, c = args.max_camber, args.location_of_max_camber, args.max_thickness, args.c

    # Adapted from https://stackoverflow.com/a/31815812
    # https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil

    def thickness(x, t, c):
        term1 = 0.2969 * (jnp.sqrt(x / c))
        term2 = -0.1260 * (x / c)
        sqr = jnp.power(x / c, 2)
        term3 = -0.3516 * sqr
        term4 = 0.2843 * sqr * (x / c)
        term5 = -0.1015 * sqr * sqr
        return 5 * t * c * (term1 + term2 + term3 + term4 + term5)

    dyc_dx = dyc_over_dx(x, args)
    th = jnp.arctan(dyc_dx)
    yt = thickness(x, t, c)
    yc = camber_line(x, m, p, c)
    return ((x - yt * jnp.sin(th), yc + yt * jnp.cos(th)), (x + yt * jnp.sin(th), yc - yt * jnp.cos(th)))


def camber_line(x, m, p, c):
    return jnp.where(
        (x >= 0) & (x <= (c * p)),
        m * (x / jnp.power(p, 2)) * (2.0 * p - (x / c)),
        m * ((c - x) / jnp.power(1 - p, 2)) * (1.0 + (x / c) - 2.0 * p),
    )


def dyc_over_dx(x, args):
    m, p, c = args.max_camber, args.location_of_max_camber, args.c
    before_transition = (((2.0 * m) / p**2) * (p - x / c)) if p != 0.0 else 0
    after_transition = ((2.0 * m) / jnp.power(1 - p, 2)) * (p - x / c)
    condition = jnp.logical_and(0 <= x, x <= (c * p))
    return jnp.where(condition, before_transition, after_transition)


def init_network_params(sizes, key, to_lists=False) -> list[tuple[list[jnp.float32], list[jnp.float32]]]:
    def random_layer_params(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        weights = scale * random.normal(w_key, (n, m))
        biases = scale * random.normal(b_key, (n,))
        return (weights.tolist(), biases.tolist()) if to_lists else (weights, biases)

    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def predict(in_array, params):
    assert isinstance(in_array, jnp.ndarray), f"Expected in_array to be of type jnp.ndarray, got {type(in_array)}"

    def relu(x):
        return jnp.maximum(0, x)

    activations = in_array
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits[0]


class LearningArgs(Tap):
    num_iters: int = 500
    step_size: float = 0.01
    layer_sizes: list[int] = [1, 10, 10, 1]
    num_integral_samples: int = 50
    seed: int = 0
    basepath: str = "./data"
    colocation_points = jnp.linspace(0, 1, 50)[1:-1]  # trim endpoints

    run: bool = False
    save_files: bool = False
    load_files: Literal[True, False, "All"] = False
    plot: bool = False

    plot_title: str | None = None
    num_epochs_to_plot: int | None = None
    save_plot: bool = False
    plot_path: str = "../div_by_zero_paper/images"
    plot_log_loss: bool = False
    plot_function_xlabel: str = "Position"
    plot_function_ylabel: str = "Prediction"


class AirfoilArgs(LearningArgs):
    # NACA6412
    max_camber: float = 0.06  # maximum camber (height) as a fraction of c
    location_of_max_camber: float = 0.4  # location of maximum camber as a fraction of c
    max_thickness: float = 0.12  # maximum thickness as a fraction of c
    c: float = 1.0  # chord length

    V_inf: float = 1.0  # freestream velocity
    alpha: float = jnp.pi / 10  # angle of attack

    plot_function_ylabel: str = "Circulation Density"


def round_sigfigs(to_round: float | list[float], n: int, latex: bool = False) -> str | list[str]:
    lifted = False
    if isinstance(to_round, float):
        lifted = True
        to_round = [to_round]
    all_rounded = []
    for r in to_round:
        rounded = f"%.{n}E" % Decimal(r)
        num, exp = rounded.split("E")
        if -1 <= int(exp) <= 1:
            rounded = float(rounded)
        elif latex:
            rounded = num + " \\cdot 10^{" + str(int(exp)) + "}"
            rounded = r"$" + rounded + r"$"
        all_rounded.append(rounded)

    if lifted:
        return all_rounded[0]
    return all_rounded


def latexify(exprs: list[str]) -> list[str]:
    latex_exprs = [r"$" + e + r"$" for e in exprs[:-2]]
    [latex_exprs.append("$\\" + e + r"$") for e in exprs[-2:]]
    return latex_exprs


def train(create_loss_fun, args) -> tuple[list[float], list[tuple[list[Any], list[Any]]]]:
    params = init_network_params(args.layer_sizes, random.key(args.seed))
    init, opt_update, get_params = jax_opt.adam(args.step_size)
    opt_state = init(params)
    key = random.PRNGKey(args.seed)
    loss_fun = create_loss_fun(key, args)
    grad_fun = jax.jit(jax.jacrev(loss_fun))
    losses = []
    singular = args.singular
    for i in range(args.num_iters):
        # We want to compute the loss with the singular integral
        # Even if we are not using the singular integral to compute the gradient
        args.singular = True
        loss = loss_fun(params)
        print(f"Iteration {i}: loss", loss)
        losses.append(loss)

        # Compute the gradient which accounts for the singular integral only if args.singular=True
        args.singular = singular
        dloss_dparam = grad_fun(params)
        flat_grad_params, _ = flatten_nn_params(dloss_dparam)
        print(f"norm(deriv): ", jnp.linalg.norm(flat_grad_params))

        opt_state = opt_update(i, dloss_dparam, opt_state)
        params = get_params(opt_state)

        print()

    return losses, params
