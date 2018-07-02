import os

import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA


def _plot_embeddings(embeddings, style="seaborn-white", title=None):
    plt.style.use(style)

    fig, ax = plt.subplots()

    for i, (x, y) in enumerate(embeddings):
        ax.scatter(x, y, color="purple")
        ax.annotate(i, xy=(x, y), fontsize=20)

    if title is not None:
        ax.set_title(title, fontsize=15)

    return fig, ax


def make_plots(params, estimator):
    source = estimator.get_variable_value(name="integer_embed/embeddings")
    projections = {
        "_".join(["projection", col]): estimator.get_variable_value(
            name="projection_{}/kernel".format(col)).T
        for col in params.label_cols}

    all_embeddings = {"source": source, **projections}

    if params.embed_dim > 2:
        tf.logging.info(
            "Embedding dimension {} requires reducing. "
            "Performing PCA to extract first 2 components.".format(
                params.embed_dim))

        pca_decomposers = {k: PCA(n_components=2) for k in all_embeddings}
        pca_decomposers = {k: pca_decomposers[k].fit(emb)
                           for k, emb in all_embeddings.items()}

        all_embeddings = {k: pca_decomposers[k].transform(emb)
                          for k, emb in all_embeddings.items()}

    all_embeddings = {col: normalize(p, norm="l2", axis=1)
                      for col, p in all_embeddings.items()}

    def get_title(col):
        return ("int2vec, dataset: {}\nembedding: {}\narchitecture: {}, "
                "embed_dim: {}").format(
            params.dataset, col, params.architecture, params.embed_dim)

    plots = {
        col: _plot_embeddings(p, title=get_title(col))
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
    img_path = os.path.join(run_config.model_dir, "imgs")
    save_ok = {
        embedding_type: _save_fig(fig=fig,
                                  path=os.path.join(img_path, embedding_type))
        for embedding_type, (fig, _) in plots.items()}

    return save_ok
