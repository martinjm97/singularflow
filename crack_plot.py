import matplotlib.pyplot as plt
import numpy as np
from tap import Tap
from matplotlib import rcParams


class CrackPlotArgs(Tap):
    h: float = 10  # Width of the strip
    a: float = 4  # Start of the crack
    b: float = 6  # End of the crack
    y: float = 5  # The arbitrary range for y

    n_zigs: int = 19  # Number of zigs in the crack

    save_files: bool = False
    path: str = "/Users/jessemichel/research/div_by_zero_project/div_by_zero_paper/images/"


def plot_crack(args):
    h, y = args.h, args.y
    a, b = args.a, args.b
    n_zigs = args.n_zigs
    lim_fs = 18
    crack_text_fs = 20
    label_fs = 20
    title_fs = 26

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.fill_betweenx(np.linspace(-y, y, 10), 0, h, color='lightgray', alpha=0.5, label='Infinite Plane Strip')

    # Plot the crack
    xs = np.linspace(0, 1, n_zigs)
    ys = [0 if int(i) % 2 == 0 else -0.1 for i in xs * n_zigs]
    plt.plot(a + (b - a) * xs, ys, color='black', label='Crack')

    ax.text(a, - 0.3, 'a', ha='center', va='top', color='black', fontsize=crack_text_fs)
    ax.text(b, - 0.3, 'b', ha='center', va='top', color='black', fontsize=crack_text_fs)

    # Set plot limits
    ax.set_xlim(0, h)
    ax.set_ylim(-y, y)

    ax.set_xticks([0, h])
    ax.set_yticks([-y, 0, y])
    # ax.set_yticks([0])
    ax.tick_params(axis='x', labelsize=lim_fs)
    ax.tick_params(axis='y', labelsize=lim_fs)

    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)

    ax.set_xlabel('x', fontsize=label_fs)
    ax.set_ylabel('y', fontsize=label_fs)
    # ax.set_title('Infinite Plane Strip with Crack', fontsize=title_fs)
    plt.tight_layout()
    if args.save_files:
        rcParams["text.usetex"] = True
        rcParams["font.family"] = "libertine"
        plt.savefig(f"{args.path}/crack_fig.pdf")
        plt.clf()
    else:
        plt.show()


if __name__ == '__main__':
    args = CrackPlotArgs().parse_args()
    plot_crack(args)
