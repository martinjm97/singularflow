import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from training import load_model


def plot_losses(train_losses, test_losses, args, ground_truth_fun=None, extra_text: str = ""):
    for losses, title in [(train_losses, "Train"), (test_losses, "Test")]:
        if args.save_plot:
            rcParams["text.usetex"] = True
            rcParams["font.family"] = "libertine"
        if args.load_files == "All" or len(losses) == 2:
            plt.figure(figsize=(7, 5))
            plt.plot(losses[0][: args.num_epochs_to_plot], label="Ours", color="green", linewidth=4)
            plt.plot(losses[1][: args.num_epochs_to_plot], label="Standard", color="blue", linestyle="--", linewidth=4)
            if ground_truth_fun is not None:
                len_losses = len(losses[0][: args.num_epochs_to_plot])
                plt.plot(
                    [0, len_losses],
                    [ground_truth_fun(args), ground_truth_fun(args)],
                    label=args.truth_title,
                    color="purple",
                    linestyle="dotted",
                    linewidth=4,
                )

            if args.plot_log_loss:
                plt.yscale("log")

            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Iteration", fontsize=32)
            plt.ylabel(f"{'Log ' if args.plot_log_loss else ''}{title} Loss", fontsize=32)
            plt.legend(frameon=False, fontsize=26)
            plt.tight_layout()
        elif args.load_files == "AllSeeds":
            if ground_truth_fun is not None:
                ground_truth_losses = []
                for seed in range(args.num_seeds):
                    args.seed = seed if title == "Train" else seed + args.num_seeds
                    ground_truth_losses.append(ground_truth_fun(args, train=(title == "Train")))
            if args.plot_log_loss:
                losses = np.log(losses)
                if ground_truth_fun is not None:
                    ground_truth_losses = np.log(ground_truth_losses)
            plt.figure(figsize=(7, 5))

            if ground_truth_fun is not None:
                len_losses = len(losses[0::2][0])
                ground_truth_mean_loss = np.mean(ground_truth_losses, axis=0)
                ground_truth_std_loss = np.std(ground_truth_losses, axis=0)
                plt.fill_between(
                    [0, len_losses],
                    2 * [ground_truth_mean_loss - ground_truth_std_loss],
                    2 * [ground_truth_mean_loss + ground_truth_std_loss],
                    color="purple",
                    alpha=0.2,
                )
                (ground_truth_handle,) = plt.plot(
                    [0, len_losses],
                    [ground_truth_mean_loss, ground_truth_mean_loss],
                    label=args.truth_title,
                    color="purple",
                    linestyle="dotted",
                    linewidth=4,
                )
            standard = losses[1::2]
            ours = losses[0::2]
            standard_mean = np.mean(standard, axis=0)
            ours_mean = np.mean(ours, axis=0)
            standard_std = np.std(standard, axis=0)
            ours_std = np.std(ours, axis=0)
            xs = np.arange(len(standard_mean))
            plt.fill_between(
                xs,
                standard_mean - standard_std,
                standard_mean + standard_std,
                color="blue",
                alpha=0.3,
            )
            plt.fill_between(xs, ours_mean - ours_std, ours_mean + ours_std, color="green", alpha=0.3)
            (ours_handle,) = plt.plot(ours_mean, label="Ours", color="green", linewidth=4)
            (standard_handle,) = plt.plot(standard_mean, label="Standard", color="blue", linestyle="--", linewidth=4)

            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel("Iteration", fontsize=32)
            plt.ylabel(f"{'Log ' if args.plot_log_loss else ''}{title} Loss", fontsize=32)
            legend_labels = ["Ours", "Standard"]
            legend_handles = [ours_handle, standard_handle]
            if ground_truth_fun is not None:
                legend_labels.append(args.truth_title)
                legend_handles.append(ground_truth_handle)
            plt.legend(frameon=False, fontsize=26, labels=legend_labels, handles=legend_handles, loc="upper right")
            plt.tight_layout()
        else:
            plt.plot(losses)
            plt.title("Loss over iterations")
        if args.save_plot:
            plt.savefig(
                f"{args.plot_path}/{args.plot_title.lower().replace(' ', '_')}{extra_text}_{title.lower()}_loss_plot.pdf"
            )
            plt.clf()
        else:
            plt.show()


def plot_functions(bounds, params, args, ground_truth=None, extra_text: str = ""):
    if args.save_plot:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "libertine"
    if args.load_files == "All":
        a, b = bounds
        singularp, standardp = params
        xs = np.linspace(a, b, 100)[1:]
        singular_model, standard_model = load_model(singularp, args), load_model(standardp, args)
        singular_nn_at_xs = jax.vmap(lambda x: singular_model(jnp.array([x])))(xs)
        standard_nn_at_xs = jax.vmap(lambda x: standard_model(jnp.array([x])))(xs)

        plt.figure(figsize=(7, 5))

        plt.plot(xs, singular_nn_at_xs, label="Ours", color="green", linewidth=4)
        plt.plot(xs, standard_nn_at_xs, label="Standard", color="blue", linestyle="--", linewidth=4)

        if ground_truth is not None:
            fun_at_xs = [ground_truth(x) for x in xs]
            if fun_at_xs[0] == float("NaN"):
                fun_at_xs[0] = None
            plt.plot(xs, fun_at_xs, label=args.truth_title, color="purple", linestyle="dotted", linewidth=4)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(args.plot_function_xlabel, fontsize=32)
        plt.ylabel(args.plot_function_ylabel, fontsize=32)
        plt.legend(frameon=False, fontsize=26)
        plt.tight_layout()
        if args.save_plot:
            plt.savefig(f"{args.plot_path}/{args.plot_title.lower().replace(' ', '_')}{extra_text}_function.pdf")
            plt.clf()
        else:
            plt.show()
    elif args.load_files == "AllSeeds":
        a, b = bounds
        singularps, standardps = params[0::2], params[1::2]
        xs = np.linspace(a, b, 100)[1:]
        singular_nn_at_xss, standard_nn_at_xss = [], []
        for singularp, standardp in zip(singularps, standardps):
            singular_model, standard_model = load_model(singularp, args), load_model(standardp, args)
            singular_nn_at_xss.append(jax.vmap(lambda x: singular_model(jnp.array([x])))(xs))
            standard_nn_at_xss.append(jax.vmap(lambda x: standard_model(jnp.array([x])))(xs))
        singular_nn_at_xs = np.mean(singular_nn_at_xss, axis=0).flatten()
        standard_nn_at_xs = np.mean(standard_nn_at_xss, axis=0).flatten()
        singular_std = np.std(singular_nn_at_xss, axis=0).flatten()
        standard_std = np.std(standard_nn_at_xss, axis=0).flatten()

        plt.figure(figsize=(7, 5))

        plt.fill_between(
            xs,
            singular_nn_at_xs - singular_std,
            singular_nn_at_xs + singular_std,
            color="green",
            alpha=0.3,
        )
        plt.plot(xs, singular_nn_at_xs, label="Ours", color="green", linewidth=4)

        plt.fill_between(
            xs,
            standard_nn_at_xs - standard_std,
            standard_nn_at_xs + standard_std,
            color="blue",
            alpha=0.3,
        )
        plt.plot(xs, standard_nn_at_xs, label="Standard", color="blue", linestyle="--", linewidth=4)

        if ground_truth is not None:
            fun_at_xs = [ground_truth(x, None) for x in xs]
            if fun_at_xs[0] == float("NaN"):
                fun_at_xs[0] = None
            plt.plot(xs, fun_at_xs, label=args.truth_title, color="purple", linestyle="dotted", linewidth=4)

        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel(args.plot_function_xlabel, fontsize=32)
        plt.ylabel(args.plot_function_ylabel, fontsize=32)
        plt.legend(frameon=False, fontsize=26)
        plt.tight_layout()
        if args.save_plot:
            plt.savefig(f"{args.plot_path}/{args.plot_title.lower().replace(' ', '_')}{extra_text}_function.pdf")
            plt.clf()
        else:
            plt.show()
