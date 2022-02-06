from turtle import width
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tueplots import fontsizes
import matplotlib.font_manager
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import sys

sys.path.insert(0, "../src/")
from eval_models import make_prediction_data, train_model_variants


from operator import sub


def get_aspect(ax):
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def scatter_hist(x1, x2, y, ax1, ax2, ax_histx1, ax_histx2, ax_histy):
    colors = ["#0c0b00", "#deb522", "#128bb5"]  # first color is black, last is red
    cm = LinearSegmentedColormap.from_list("Custom", colors, N=50, gamma=0.8)

    x1_sort_idx = np.abs(x1 - y).argsort()
    x2_sort_idx = np.abs(x2 - y).argsort()

    # no labels
    ax_histx1.tick_params(axis="x", labelbottom=False)
    ax_histx2.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    vmin, vmax = 0, 2

    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    sc1 = ax1.scatter(
        x1[x1_sort_idx[::-1]],
        y[x1_sort_idx[::-1]],
        c=np.abs(x1 - y)[x1_sort_idx[::-1]],
        norm=normalize,
        cmap=cm,
        s=3,
    )
    ax2.scatter(
        x2[x2_sort_idx[::-1]],
        y[x2_sort_idx[::-1]],
        c=np.abs(x2 - y)[x2_sort_idx[::-1]],
        norm=normalize,
        cmap=cm,
        s=3,
    )

    cbar = plt.colorbar(
        sc1, use_gridspec=True, ticks=[0, 0.5, 1, 1.5, 2], fraction=0.2, pad=0.0, location='right'
    )
    cbar.ax.set_yticklabels(["0", "0.5", "1", "1.5", "$>2$"])
    cbar.ax.set_ylabel("Absolute Error")

    ax1.set_xlim([1.0, 9.5])
    ax2.set_xlim([1.0, 9.5])
    ax1.set_ylim([1.0, 9.5])
    ax2.set_ylim([1.0, 9.5])

    ax_histy.set_ylim([1.0, 9.5])

    ax1.set_xticks(ticks=[2, 4, 6, 8], labels=None)
    ax2.set_xticks(ticks=[2, 4, 6, 8], labels=None)
    ax1.set_yticks(ticks=[2, 4, 6, 8], labels=None)
    ax2.set_yticks([])

    ax1.set_ylabel("True Ratings")
    ax1.set_xlabel("Logistic Regression Pred.")
    ax2.set_xlabel("ReLU-Net-6 Pred.")

    bins = 25

    #ax_histx1 = sns.distplot(x1, ax=ax_histx1, color="#deb522", bins=bins)
    #ax_histx2 = sns.distplot(x2, ax=ax_histx2, color="#128bb5", bins=bins)
    ax_histx1 = sns.distplot(x1, ax=ax_histx1, color="#deb522", bins=bins, kde_kws={"linewidth":0.9})
    ax_histx2 = sns.distplot(x2, ax=ax_histx2, color="#128bb5", bins=bins, kde_kws={"linewidth":0.9})
    ax_histy = sns.distplot(y, ax=ax_histy, color="#000000", vertical=True, bins=bins, kde_kws={"linewidth":0.9})

    # ax_histx.hist(x, bins=bins, color = "#deb522")

    ax_histy.set_xlim(ax_histx1.get_ylim())

    for a in [ax_histx1, ax_histx2]:
        a.set_yticks([])
        a.set_ylabel("")

        a.spines["top"].set_visible(False)
        a.spines["left"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["bottom"].set_visible(False)

    ax_histy.set_yticks([])
    ax_histy.set_xticks([])
    ax_histy.set_xlabel("")

    ax_histy.spines["top"].set_visible(False)
    ax_histy.spines["right"].set_visible(False)
    ax_histy.spines["bottom"].set_visible(False)

    return ax1, ax2, ax_histx1, ax_histx2, ax_histy


def make_plot():
    # Options
    params = {
        "text.latex.preamble": r"\usepackage{lmodern}",
        "text.usetex": True,
        "font.size": 9,
        "font.family": "sans",
    }
    plt.rcParams.update(params)
    plt.rcParams.update(fontsizes.neurips2021())

    from_memory = False

    if from_memory:

        x1 = np.load("../dat/data_fig1/pred_lr.npy")
        x2 = np.load("../dat/data_fig1/pred_relu.npy")
        y = np.load("../dat/data_fig1/true_ratings.npy")

    else:
        data = pd.read_csv("../dat/data_clean.csv", dtype={5: "object", 16: "object"})
        (
            (train_set_normalized, train_targets),
            (val_set_normalized, val_targets),
            (test_set_normalized, test_targets),
        ) = make_prediction_data(data)

        result_dict = train_model_variants(
            train_set_normalized,
            train_targets,
            val_set_normalized,
            val_targets,
            test_set_normalized,
            test_targets,
        )

        x1 = result_dict["MSE"][2]
        x2 = result_dict["RELU6"][2]
        y = result_dict["ground_truth"]

    # start with a square Figure
    # fig = plt.figure(figsize=(4, 4)) 5.499999861629998, 2.266124568404705
    size = set_size(397.48499, fraction=1, subplots=(1.5, 2.5))
    _, ax = plt.subplots(
        2,
        3,
        gridspec_kw={"height_ratios": [0.0812, 0.3], "width_ratios": [0.3, 0.3, 0.105]},
        figsize=(size[0] * 0.8, 3.0 * 0.8),
    )

    # ax[1, 0].set_aspect(1, anchor="SE")
    # ax[1, 1].set_aspect(1, anchor="SE")
    ax[1, 1].set_yticks([2, 4, 6, 8], minor=True)
    ax[1, 1].tick_params(axis='y', which='minor',length=0)

    ax[1, 0].grid(True, alpha=0.4, which="both")#,  linewidth=0.5)
    ax[1, 1].grid(True, alpha=0.4, which="major")#, linewidth=0.5)
    ax[1, 1].grid(True, alpha=0.4, which="minor")#, linewidth=0.5)
    ax[1, 1].yaxis.grid(True)

    ax[1, 0].set_axisbelow(True)
    ax[1, 1].set_axisbelow(True)

    ax[1, 0].get_shared_x_axes().join(ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1])
    ax[1, 0].get_shared_y_axes().join(ax[1, 0], ax[1, 1])
    ax[0, 0].get_shared_y_axes().join(ax[0, 0], ax[0, 1])

    ax[0, 2].axis("off")

    scatter_hist(x1, x2, y, ax[1, 0], ax[1, 1], ax[0, 0], ax[0, 1], ax[1, 2])

    # plt.axis('equal')

    plt.subplots_adjust(wspace=0, hspace=0)

    bbox = ax[0, 0].get_window_extent().transformed(_.dpi_scale_trans.inverted())
    print("normal", bbox.width, bbox.height)
    bbox = ax[1, 2].get_window_extent().transformed(_.dpi_scale_trans.inverted())
    print("y", bbox.width, bbox.height)


    # The gfx folder needs to exist
    # 397.48499p
    plt.savefig(
        "../doc/gfx/fig1.pdf",
        # This is simple recomendation for publication plots
        dpi=1000,
        # Plot will occupy all of available space
        bbox_inches="tight",
        pad_inches=0.0,
    )

    plt.savefig(
        "../doc/gfx/fig1.png",
        bbox_inches="tight",
        pad_inches=0.0,
        facecolor="white",
        dpi=1000,
    )


if __name__ == "__main__":
    make_plot()
