import os
import pickle
import time
from functools import partial
from typing import Literal

import jax.numpy as jnp
import optax
from flax import nnx
from jax import random
from tap import Tap


class LearningArgs(Tap):
    num_iters: int = 1000
    step_size: float = 0.01
    layer_sizes: list[int] = [1, 100, 100, 1]
    num_integral_samples: int = 50
    num_test_integral_samples: int = 1000
    seed: int = 0
    basepath: str = "./data"
    colocation_points = jnp.linspace(0, 1, 50)[1:-1]  # trim endpoints
    singular: bool = True
    num_seeds: int = 1

    run: bool = False
    save_files: Literal[True, False, "AllSeeds"] = False
    load_files: Literal[True, False, "All", "AllSeeds"] = False
    plot: bool = False

    path: str | None = None

    plot_title: str | None = None
    num_epochs_to_plot: int | None = None
    save_plot: bool = False
    plot_path: str = "../div_by_zero_paper/images"
    plot_log_loss: bool = False
    plot_function_xlabel: str = "Position"
    plot_function_ylabel: str = "Prediction"
    truth_title: str = "Math Approx."


class MLP(nnx.Module):

    def __init__(self, dims: list[int], rngs: nnx.Rngs):
        self.linears = [nnx.Linear(feat_in, feat_out, rngs=rngs) for feat_in, feat_out in zip(dims, dims[1:])]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.linears[:-1]:
            x = nnx.gelu(layer(x))
        return self.linears[-1](x)


@partial(nnx.jit, static_argnums=[2, 3, 4, 5])
def train_step(model: nnx.Module, optimizer, loss_fn, test_loss_fn, grad_fn, args: LearningArgs):
    singular = args.singular
    # Always compute the loss with the singular integral
    args.singular = True
    train_loss = loss_fn(model)
    test_loss = test_loss_fn(model)

    # The gradient is computed with the singular integral only if args.singular=True
    args.singular = singular
    grads = grad_fn(model)
    optimizer.update(grads)
    return train_loss, test_loss


def train(model, create_loss_fun, key, args: LearningArgs) -> tuple[list[float], list[float], MLP]:
    optimizer = nnx.Optimizer(model, optax.adam(args.step_size))
    loss_fn = create_loss_fun(key, args)
    grad_fn = nnx.grad(loss_fn)

    key = random.PRNGKey(args.seed + args.num_seeds)
    test_loss_fn = create_loss_fun(key, args, train=False)
    losses = []
    test_losses = []
    for i in range(args.num_iters):
        train_loss, test_loss = train_step(model, optimizer, loss_fn, test_loss_fn, grad_fn, args)
        print(f"Iteration {i}: train loss", train_loss, "test loss", test_loss)
        losses.append(train_loss)
        test_losses.append(test_loss)
    return losses, test_losses, model


def load_model(params, args: LearningArgs):
    model = MLP(args.layer_sizes, nnx.Rngs(random.PRNGKey(args.seed)))
    nnx.update(model, params)
    return model


def run_single_seed(create_loss_fun, args: LearningArgs):
    key = random.PRNGKey(args.seed)
    start = time.time()
    model = MLP(args.layer_sizes, nnx.Rngs(key))
    train_losses, test_losses, model = train(model, create_loss_fun, key, args)
    print(f"Training time: {time.time() - start} seconds")
    params = nnx.state(model, nnx.Param)
    return train_losses, test_losses, params


def save_single_run(path, args, train_losses, test_losses, params):
    if not os.path.exists(path):
        os.makedirs(path)

    args.save(f"{path}/singular_{args.singular}_args.json")
    with open(f"{path}/singular_{args.singular}_params.npy", "wb") as f:
        pickle.dump(params, f)
    with open(f"{path}/singular_{args.singular}_train_losses.npy", "wb") as f:
        jnp.save(f, jnp.array(train_losses))
    with open(f"{path}/singular_{args.singular}_test_losses.npy", "wb") as f:
        jnp.save(f, jnp.array(test_losses))


def run(path, train, create_loss_fun, args: LearningArgs):
    if args.run:
        assert args.num_seeds > 0, "num_seeds must be greater than 0."
        if args.num_seeds == 1:
            all_train_losses, all_test_losses, all_params = run_single_seed(create_loss_fun, args)
        else:
            all_train_losses, all_test_losses, all_params = [], [], []
            for seed in range(args.num_seeds):
                for singular in [True, False]:
                    args.singular = singular
                    args.seed = seed
                    train_losses, test_losses, p = run_single_seed(create_loss_fun, args)
                    all_train_losses.append(train_losses)
                    all_test_losses.append(test_losses)
                    all_params.append(p)

    match args.save_files:
        case False:
            pass
        case True:
            assert args.run, f"Must run to save files."
            save_single_run(path, args, all_train_losses, all_test_losses, all_params)
        case "AllSeeds":
            for seed in range(args.num_seeds):
                for singular in [True, False]:
                    args.singular = singular
                    args.seed = seed
                    save_single_run(
                        f"{path}/seed_{seed}",
                        args,
                        train_losses=all_train_losses[2 * seed + (not singular)],
                        test_losses=all_test_losses[2 * seed + (not singular)],
                        params=all_params[2 * seed + (not singular)],
                    )

    match args.load_files:
        case True:
            with open(f"{path}/singular_{args.singular}_params.npy", "rb") as f:
                all_params = pickle.load(f)
            with open(f"{path}/singular_{args.singular}_train_losses.npy", "rb") as f:
                all_train_losses = jnp.load(f)
            with open(f"{path}/singular_{args.singular}_test_losses.npy", "rb") as f:
                all_test_losses = jnp.load(f)
        case "All":
            all_params, all_train_losses, all_test_losses = [], [], []
            for singular in [True, False]:
                with open(f"{path}/singular_{singular}_params.npy", "rb") as f:
                    all_params.append(pickle.load(f))
                with open(f"{path}/singular_{singular}_train_losses.npy", "rb") as f:
                    all_train_losses.append(jnp.load(f))
                with open(f"{path}/singular_{singular}_test_losses.npy", "rb") as f:
                    all_test_losses.append(jnp.load(f))
        case "AllSeeds":
            all_params, all_train_losses, all_test_losses = [], [], []
            for seed in range(args.num_seeds):
                for singular in [True, False]:
                    with open(f"{path}/seed_{seed}/singular_{singular}_params.npy", "rb") as f:
                        all_params.append(pickle.load(f))
                    with open(f"{path}/seed_{seed}/singular_{singular}_train_losses.npy", "rb") as f:
                        all_train_losses.append(jnp.load(f))
                    with open(f"{path}/seed_{seed}/singular_{singular}_test_losses.npy", "rb") as f:
                        all_test_losses.append(jnp.load(f))
        case False:
            pass

    if args.run or args.load_files:
        return all_train_losses, all_test_losses, all_params
    else:
        return None, None, None
