import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def hist(vals, title=None, save=None, xlim=None):
    """vals: num list"""
    ax = sns.distplot(vals, kde=False)
    if xlim is not None:
        ax.set(xlim=(0, 230))
    if title is not None:
        ax.set_title(title)
    if save is not None:
        ax.figure.savefig(save)


def boxplot(data, aes, title=None, save=None, figsize=(10, 4)):
    """
    aes: dict(x='x-map', y='y-map', col='col-map')
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(data=data, x=aes["x"], y=aes["y"], hue=aes["col"], ax=ax)
    if title is not None:
        ax.set_title(title)
    if save is not None:
        fig.savefig(save)


def residuals(preds, truth, title=None, save=None):
    """
    truth, preds, list of nums
    """
    df = pd.DataFrame(
        dict(truth=truth, preds=preds, resids=[t - p for t, p in zip(truth, preds)])
    )
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    g = sns.scatterplot(x="preds", y="truth", data=df, ax=ax[0])
    g.plot(
        [min(preds), max(preds)], [min(preds), max(preds)], color="gray", linestyle="--"
    )
    g.set_title("Prediction")
    g = sns.scatterplot(x="preds", y="resids", data=df, ax=ax[1])
    g.plot([min(preds), max(preds)], [0, 0], color="gray", linestyle="--")
    g.set_title("Residual Plot")
    g = sns.distplot(df["resids"], ax=ax[2])
    g.set_title("Residual Distribution")
    if title is not None:
        plt.subplots_adjust(top=0.8)
        fig.suptitle(title, fontsize=16)
    if save is not None:
        fig.savefig(save)


def barplot(data, aes, title=None, save=None, aspect=1.5, height=4, dodge=False):
    """
    aes: dict(x='x-map', y='y-map', hue='col-map')
    """
    g = sns.catplot(
        **aes,
        data=data,
        dodge=dodge,
        legend=False,
        kind="bar",
        aspect=aspect,
        height=height
    )
    plt.legend(loc="lower left")
    axs = g.axes.flatten()
    if title is not None:
        axs[0].set_title(title)
    if save is not None:
        axs[0].figure.savefig(save)


def train_progress(data, aes, title=None, save=None, aspect=3, height=4):
    """
    aes: dict(x='x-map', loss=['Train-Loss', 'Val-Loss'], perf=['Train-Rsq', 'Val-Rsq'])
    """
    df = data.melt(id_vars=aes["x"], value_vars=aes["loss"]).append(
        data.melt(id_vars=aes["x"], value_vars=aes["perf"]), ignore_index=True
    )
    loss = aes["loss"][0].split("-")[1]
    perf = aes["perf"][0].split("-")[1]
    df["metric"] = [loss] * int(len(df) / 2) + [perf] * int(len(df) / 2)
    df["part"] = df["variable"].apply(lambda d: d.split("-")[0])

    g = sns.FacetGrid(
        df, row="metric", aspect=3, height=4, sharey=False, legend_out=False
    )
    g = g.map_dataframe(sns.lineplot, x=aes["x"], y="value", hue="part").add_legend()
    if title is not None:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
    if save is not None:
        g.fig.savefig(save)


def scatterpair(data, aes, title=None, save=None, aspect=1, height=5):
    """
    aes: dict(x='x-map', y='y-map', hue='color-map', col='column-map')
    """
    df_map = {k: v for k, v in aes.items() if k != "col"}
    g = (
        sns.FacetGrid(
            data=data,
            col=aes["col"],
            aspect=1,
            height=5,
            sharey=False,
            sharex=False,
            legend_out=False,
        )
        .map_dataframe(sns.scatterplot, **df_map)
        .add_legend()
    )
    if title is not None:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
    if save is not None:
        g.fig.savefig(save)


def distpair(data, aes, title=None, save=None, aspect=1, height=5):
    """
    aes: dict(a='a-map', col='column-map')
    """
    g = (
        sns.FacetGrid(
            data=data,
            col=aes["col"],
            aspect=1,
            height=5,
            sharey=False,
            sharex=False,
            legend_out=False,
        )
        .map(sns.distplot, aes["a"])
        .add_legend()
    )
    if title is not None:
        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
    if save is not None:
        g.fig.savefig(save)


def heatmap(df, title=None, save=None, figsize=(9, 7), cmap="YlGnBu", annot=True):
    """
    df of heatmap, indexes and columns are axis labels
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=annot, cmap=cmap, ax=ax)
    if title is not None:
        plt.subplots_adjust(top=0.95)
        fig.suptitle(title, fontsize=16)
    if save is not None:
        fig.savefig(save)


def multiconf(m, labels, title=None, save=None):
    """
    m: np.array confusion matrix
    labels: ordered list of label names (axis labels)
    """
    df = pd.DataFrame(m)
    df.columns = labels
    df.index = labels
    heatmap(df, title=title, save=save)


def predictive_vals(metrics, title=None, save=None):
    """
    metrics: list of dicts from `utils.evaluation.get_confusion_metrics`
    """
    df = pd.DataFrame(metrics).melt(id_vars=["label"], value_vars=["PPV", "TPR"])
    barplot(
        df,
        dict(x="label", y="value", hue="variable"),
        dodge=True,
        title=title,
        save=save,
    )


def lineplot(data, aes, title=None, save=None, line=None, figsize=(8, 8)):
    """
    aes: dict(x='x-map', y='y-map', hue='hue-map')
    line: dict(x=[x0, x1], y=[y0, y1]) for drawing straight line
    """
    fig, ax = plt.subplots(figsize=figsize)
    g = sns.lineplot(data=data, x=aes["x"], y=aes["y"], hue=aes["hue"], ax=ax)
    if line is not None:
        g.plot(line["x"], line["y"], color="gray", linestyle="--")
    if title is not None:
        g.set_title(title)
    if save is not None:
        fig.savefig(save)
