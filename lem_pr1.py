from datetime import datetime

import jax
import jax.numpy as jnp

from singular_integrate import get_average_value, singular_integrate
from plotting import plot_functions, plot_losses
from training import LearningArgs, run, train
from flax import nnx


class LEMArgs(LearningArgs):
    a: float = 4  # crack start
    b: float = 6  # crack end
    kappa: float = 3 - 4 * 0.25  # elastic constant, where = 0.25 is the poisson ratio for granite
    mu: float = 30  # Lamé second parameter similar to the shear modulus
    px: float = 1  # surface traction (constant load)
    #     colocation_points = jnp.linspace(4, 6, 50)[1:-1]  # trim endpoints
    colocation_points = jnp.linspace(4, 6, 50)[1:-1]  # trim endpoints

    plot_title: str = "Crack Problem"
    plot_log_loss = True
    plot_function_ylabel = "Crack Displacement"


def create_loss_fun(key, args, train=True):
    num_samples = args.num_integral_samples if train else args.num_test_integral_samples

    a, b, kappa, mu = args.a, args.b, args.kappa, args.mu
    bounds = (a, b)

    def wrapper(model):
        graph_def, state = nnx.split(model)

        # the nn models V(x), the crack opening displacement
        def numer1(t, state):
            model = nnx.merge(graph_def, state)
            return jnp.squeeze(model(jnp.expand_dims(t, 0)))

        def integrand1(x, t, state):
            return numer1(t, state) / (x - t) ** 2

        def integrate1(x):
            if args.singular:
                return singular_integrate(numer1, 2, bounds, key, num_samples, state, x)
            else:
                samples = jax.random.uniform(key, (num_samples,), minval=bounds[0], maxval=bounds[1])
                return (bounds[1] - bounds[0]) * get_average_value(lambda x, t: integrand1(x, t, state), x, samples)

        def integrand2(x, t, state):
            # continuous function simplified from equation 63-64 of
            # On the Solution of Integral Equations with Strongly Singular Kernels
            # https://ntrs.nasa.gov/api/citations/19860017516/downloads/19860017516.pdf
            return -numer1(t, state) / (t + x) ** 2

        def integrate2(x):
            samples = jax.random.uniform(key, (num_samples,), minval=bounds[0], maxval=bounds[1])
            return (bounds[1] - bounds[0]) * get_average_value(lambda x, t: integrand2(x, t, state), x, samples)

        def loss_fun_at_pnt(x):
            rhs = -jnp.pi * ((1 + kappa) / (2 * mu)) * args.px
            loss = (integrate1(x) + integrate2(x) - rhs) ** 2
            return loss

        return jnp.sum(jax.vmap(loss_fun_at_pnt)(args.colocation_points))

    return wrapper


if __name__ == "__main__":
    args = LEMArgs(explicit_bool=True).parse_args()

    if args.path is None:
        args.path = f"{args.basepath}/{datetime.today().strftime('%B_%d_%Y').lower()}/pr1"
    else:
        args.path = f"{args.basepath}/{args.path}/pr1"

    train_losses, test_losses, params = run(args.path, train, create_loss_fun, args)

    if args.plot:
        assert args.run or args.load_files, "Must run or load files to plot."
        plot_losses(train_losses, test_losses, args)
        plot_functions((args.a, args.b), params, args)
