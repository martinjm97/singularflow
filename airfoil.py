from datetime import datetime

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from flax import nnx

from singular_integrate import get_average_value, singular_integrate, get_samples
from plotting import plot_functions, plot_losses
from training import LearningArgs, run, train


class AirfoilArgs(LearningArgs):
    # NACA6412
    max_camber: float = 0.06  # maximum camber (height) as a fraction of c
    location_of_max_camber: float = 0.4  # location of maximum camber as a fraction of c
    max_thickness: float = 0.12  # maximum thickness as a fraction of c
    c: float = 1.0  # chord length

    V_inf: float = 1.0  # freestream velocity
    alpha: float = jnp.pi / 10  # angle of attack

    plot_function_ylabel: str = "Circulation Density"
    plot_title: str = "Airfoil Equation"
    # debiased: bool = False
    # plot_log_loss = True


def dyc_over_dx(x, args: AirfoilArgs):
    m, p, c = args.max_camber, args.location_of_max_camber, args.c
    before_transition = (((2.0 * m) / p**2) * (p - x / c)) if p != 0.0 else 0
    after_transition = ((2.0 * m) / jnp.power(1 - p, 2)) * (p - x / c)
    condition = jnp.logical_and(0 <= x, x <= (c * p))
    return jnp.where(condition, before_transition, after_transition)


def loss_fun_at_pnt(fun_of_s, s, key):
    rhs = args.V_inf * (jnp.sin(args.alpha) - dyc_over_dx(s, args) * jnp.cos(args.alpha))
    return (0.5 / jnp.pi * fun_of_s(s, key) - rhs) ** 2


# def loss_fun_at_pnt_debiased(fun_of_s, s, key):
#     rhs = args.V_inf * (jnp.sin(args.alpha) - dyc_over_dx(s, args) * jnp.cos(args.alpha))
#     keys = jax.random.split(key, 2)
#     est1 = fun_of_s(s, keys[0])
#     est2 = fun_of_s(s, keys[1])
#     return (0.5 / jnp.pi) ** 2 * est1 * est2 - 2 * rhs * (0.5 / jnp.pi) * est1 + rhs**2


def create_loss_fun(key, args: AirfoilArgs, train=True):
    num_samples = args.num_integral_samples if train else args.num_test_integral_samples

    def wrapper(model):
        graph_def, state = nnx.split(model)
        bounds = (0, args.c)
        pow = 1

        def numer(x, state):
            model = nnx.merge(graph_def, state)
            return jnp.squeeze(model(jnp.expand_dims(x, 0)))

        def integrand(s, x, state):
            """Integrand in the airfoil equation

            :param s: colocation point
            :param x: variable of integration
            :return: integrand
            """
            return numer(x, state) / (x - s)

        def integrate(integrand, s, key, state):
            """Integrate the integrand over the airfoil

            :param integrand: integrand
            :param s: colocation point
            :return: integral
            """

            if args.singular:
                return singular_integrate(numer, pow, bounds, key, num_samples, state, s)

            else:
                samples = jax.random.uniform(key, (num_samples,), minval=bounds[0], maxval=bounds[1])
                return (bounds[1] - bounds[0]) * get_average_value(lambda s, x: integrand(s, x, state), s, samples)

        integrand_at_pnt = lambda s, key: -integrate(integrand, s, key, state)

        # if args.debiased:
        #     lf = lambda s, key: loss_fun_at_pnt_debiased(integrand_at_pnt, s, key)
        # else:
        lf = lambda s, key: loss_fun_at_pnt(integrand_at_pnt, s, key)

        return jnp.sum(jax.vmap(lf, in_axes=(0, None))(args.colocation_points, key))

    return wrapper


def ground_truth(x, _):
    return lax.cond(
        x == 0, lambda x: float("NaN"), lambda x: 2 * args.alpha * args.V_inf * jnp.sqrt((args.c - x) / x), x
    )


def ground_truth_loss(args, train=True):
    num_samples = args.num_integral_samples if train else args.num_test_integral_samples
    key = jax.random.PRNGKey(args.seed)

    def integrate(s):
        def singular_int(s, key):
            # Set theta=0.0 as a dummy. It won't be used anyways.
            return -singular_integrate(ground_truth, 1, (0, args.c), key, num_samples, 0.0, s)

        return loss_fun_at_pnt(singular_int, s, key)

    return jnp.sum(jax.vmap(integrate)(args.colocation_points))


if __name__ == "__main__":
    args = AirfoilArgs(explicit_bool=True).parse_args()

    naca_text = f"naca{str(args.max_camber)[-1]}{str(args.location_of_max_camber)[-1]}{str(args.max_thickness)[-2:]}"
    if args.path is None:
        args.path = f"{args.basepath}/{datetime.today().strftime('%B_%d_%Y').lower()}/airfoil_{naca_text}"
    else:
        args.path = f"{args.basepath}/{args.path}/airfoil_{naca_text}"

    train_losses, test_losses, params = run(args.path, train, create_loss_fun, args)

    if args.plot:
        assert args.run or args.load_files, "Must run or load files to plot."
        plot_losses(train_losses, test_losses, args, ground_truth_loss, extra_text=f"_{naca_text}")
        plot_functions((0, args.c), params, args, ground_truth, extra_text=f"_{naca_text}")
