from datetime import datetime

import jax
import jax.numpy as jnp

from singular_integrate import get_average_value, singular_integrate
from plotting import plot_functions
from plotting import plot_losses
from training import LearningArgs
from training import run
from training import train
from flax import nnx


class LEMArgs(LearningArgs):
    plot_title: str = "Crack Problem (Starr)"
    plot_function_ylabel: str = "Crack Displacement"
    truth_title: str = "Ground Truth"
    plot_log_loss = True


def create_loss_fun_pnt(integrate1, integrate2):
    def loss_fun_at_pnt(s):
        rhs = 4 * s - 2 * jnp.sqrt(s + s**2)
        loss = (integrate1(s) + integrate2(s) - rhs) ** 2
        return loss

    return loss_fun_at_pnt


def create_loss_fun(key, args, train=True):
    num_samples = args.num_integral_samples if train else args.num_test_integral_samples

    def wrapper(model):
        graph_def, state = nnx.split(model)
        bounds = (0, 1)

        def numer1(x, state):
            model = nnx.merge(graph_def, state)
            return -jnp.squeeze(model(jnp.expand_dims(x, 0)))

        def integrand1(s, x, state):
            return numer1(x, state) / (x - s)

        def integrate1(s):
            if args.singular:
                return singular_integrate(numer1, 1, bounds, key, num_samples, state, s)
            else:
                samples = jax.random.uniform(key, (num_samples,), minval=bounds[0], maxval=bounds[1])
                return (bounds[1] - bounds[0]) * get_average_value(lambda s, x: integrand1(s, x, state), s, samples)

        def integrand2(s, x, state):
            return -numer1(x, state) / (x + s)

        def integrate2(s):
            samples = jax.random.uniform(key, (num_samples,), minval=bounds[0], maxval=bounds[1])
            return (bounds[1] - bounds[0]) * get_average_value(lambda s, x: integrand2(s, x, state), s, samples)

        return jnp.sum(jax.vmap(create_loss_fun_pnt(integrate1, integrate2))(args.colocation_points))

    return wrapper


def ground_truth(x, theta):
    return (2 / jnp.pi) * jnp.sqrt(x - x**2)


def ground_truth_loss(args, train=True):
    num_samples = args.num_integral_samples if train else args.num_test_integral_samples
    key = jax.random.PRNGKey(args.seed)
    bounds = (0, 1)

    def integrate1(s):
        # Using s as a dummy state parameter since it's not used by ground_truth
        return singular_integrate(
            lambda x, theta: -ground_truth(x, theta), 1, bounds, key, args.num_integral_samples, None, s
        )

    def integrate2(s):
        samples = jax.random.uniform(key, (args.num_integral_samples,), minval=bounds[0], maxval=bounds[1])
        return (bounds[1] - bounds[0]) * get_average_value(lambda s, x: ground_truth(x, None) / (x + s), s, samples)

    func = create_loss_fun_pnt(integrate1, integrate2)

    return jnp.sum(jax.vmap(func)(args.colocation_points))


if __name__ == "__main__":
    args = LEMArgs(explicit_bool=True).parse_args()

    if args.path is None:
        args.path = f"{args.basepath}/{datetime.today().strftime('%B_%d_%Y').lower()}/starr"
    else:
        args.path = f"{args.basepath}/{args.path}/starr"

    train_losses, test_losses, params = run(args.path, train, create_loss_fun, args)

    if args.plot:
        assert args.run or args.load_files, "Must run or load files to plot."
        bounds = (0, 1)
        plot_losses(train_losses, test_losses, args, ground_truth_loss)
        plot_functions(bounds, params, args, ground_truth)
