import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def image(img, cmap="gray", figsize=(20, 10)):
    """Plot image"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    return fig, ax


def segmentation(
    img,
    segmentation,
    title=None,
    save=None,
    figsize=(20, 10),
    linewidth=2,
    edgecolor="red",
):
    """Plots Image with segmentation as rectangles overlayed"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    for i in range(len(segmentation)):
        s = segmentation[i]
        s.calculate_properties()
        cent = s.centroid
        patch = mpatches.Rectangle(
            (s.xrange[0], s.yrange[0]),
            s.xdiam,
            s.ydiam,
            fill=False,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
        ax.add_patch(patch)
    if title is not None:
        fig.suptitle(title, size=20)
    ax.set_axis_off()
    plt.tight_layout()
    fig.subplots_adjust(top=0.95)
    if save is not None:
        plt.savefig(save)
    plt.show()
