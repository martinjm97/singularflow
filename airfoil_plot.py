import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.tree_util import tree_unflatten, tree_structure
import pickle
from datetime import datetime
from matplotlib import rcParams

from utils import camber_line, AirfoilArgs, naca4


def plot_airfoil(args: AirfoilArgs):
    plt.figure(figsize=(6, 2))
    max_camber, location_of_max_camber, max_thickness, c = (
        args.max_camber,
        args.location_of_max_camber,
        args.max_thickness,
        args.c,
    )
    x = jnp.linspace(0, 1, 10000)
    airfoil_coordinates = naca4(x, args)
    upper_surface = airfoil_coordinates[0]
    lower_surface = airfoil_coordinates[1]
    color = "lightskyblue"
    plt.plot(x, lower_surface[1], color)
    plt.plot(x, upper_surface[1], color)
    plt.fill_between(x, lower_surface[1], upper_surface[1], color=color, alpha=0.5)

    if args.plot_chord:
        # Add chord line
        plt.plot([0, 1], [-0.06, -0.06], color="#6A5ACD", linestyle="--", linewidth=2, alpha=0.5)

    plt.xticks([])
    plt.yticks([])

    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # plt.plot(x, camber_line(x, max_camber, location_of_max_camber, c), "gray")
    plt.axis("equal")
    plt.ylim(-0.05, 0.15)
    if args.save_files:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "libertine"
        naca_text = (
            f"naca{str(args.max_camber)[-1]}{str(args.location_of_max_camber)[-1]}{str(args.max_thickness)[-2:]}"
        )
        print(f"Saving to {args.path}/airfoil_{naca_text}_fig.pdf")
        plt.savefig(f"{args.path}/airfoil_{naca_text}_fig.pdf")
    else:
        plt.show()


class PlotterArgs(AirfoilArgs):
    save_files: bool = False
    path: str = "/Users/jessemichel/research/div_by_zero_project/div_by_zero_paper/images/"
    plot_chord: bool = False


if __name__ == "__main__":
    args = PlotterArgs(explicit_bool=True).parse_args()

    plot_airfoil(args)
