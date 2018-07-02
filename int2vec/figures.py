import os

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import normalize


def _plot_embeddings(embeddings, style="seaborn-white", title=None):
    plt.style.use(style)

    fig, ax = plt.subplots()

    for i, (x, y) in enumerate(embeddings):
        ax.scatter(x, y, color="purple")
        ax.annotate(i, xy=(x, y), fontsize=20)

    if title is not None:
        ax.set_title(title, fontsize=30)

    return fig, ax


def make_plots(params, estimator):
    embeddings = estimator.get_variable_value(name="integer_embed/embeddings")
    projections = {
        "_".join(["projection", col]): estimator.get_variable_value(
            name="projection_{}/kernel".format(col)).T
        for col in params.label_cols}

    all_embeddings = {"embeddings": embeddings, **projections}
    all_embeddings = {col: normalize(p, norm="l2", axis=1)
                      for col, p in all_embeddings.items()}

    plots = {
        col: _plot_embeddings(
            p,
            title="int2vec skipgram integer {}".format(col))
        for col, p in all_embeddings.items()}

    return plots


def _save_fig(fig, path, verbose=True):
    dirname = os.path.dirname(path)

    if not os.path.exists(dirname):
        os.mkdir(dirname)

    save_ok = fig.savefig(path, dpi=300, figsize=(10, 10), bbox_inches="tight")

    if verbose:
        tf.logging.info("Saved figure to {}".format(path))

    return save_ok


def save_plots(plots, run_config):
    img_path = os.path.join(run_config.model_dir, "img")
    save_ok = {
        embedding_type: _save_fig(fig=fig,
                                  path=os.path.join(img_path, embedding_type))
        for embedding_type, (fig, _) in plots.items()}

    return save_ok
