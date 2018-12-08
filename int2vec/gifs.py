import os

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from sklearn.preprocessing import normalize
from tensorflow.contrib.framework.python.framework import checkpoint_utils


def flatten(x):
    return [item for sublist in x for item in sublist]


def _get_checkpointed_vars(checkpoints, name, fn=lambda x: x):
    return {k: fn(checkpoint_utils.load_variable(checkpoint_dir=k, name=name))
            for k in checkpoints}


def get_all_embeddings(params, estimator):
    model_dir = estimator.model_dir

    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    checkpoints = checkpoint_state.all_model_checkpoint_paths

    source = _get_checkpointed_vars(checkpoints=checkpoints,
                                    name="integer_embed/embeddings")

    projections = {
        "_".join(["projection", col]): _get_checkpointed_vars(
            checkpoints=checkpoints,
            name="projection_{}/kernel".format(col),
            fn=lambda x: x.T)
        for col in params.label_cols}

    all_embeddings = {"source": source, **projections}

    all_embeddings = {col: {k: normalize(v, norm="l2", axis=1)
                            for k, v in p.items()}
                      for col, p in all_embeddings.items()}

    return all_embeddings, checkpoints


def make_gif(params, estimator):
    tf.logging.info("Building gif.")

    tf.logging.info("Loading all embeddings.")
    all_embeddings, checkpoints = get_all_embeddings(params, estimator)

    fig, ax = plt.subplots(ncols=len(all_embeddings),
                           sharex='all', sharey='all')
    fig.suptitle('Int2Vec\narchitecture: {}, dataset: {}, embed_dim: {}'.format(
        params.architecture, params.dataset, params.embed_dim))

    ax_embedding_map = {k: ax[i] for i, k in enumerate(all_embeddings)}

    ckpt_0 = checkpoints[0]
    checkpoint_label = fig.text(0.5, 0.04,
                                'Checkpoint: {}'.format(ckpt_0), ha='center')

    def get_sc_annotations(embedding_type):
        embedding_ax = ax_embedding_map[embedding_type]
        embedding_ax.set_xlim(-1, 1)
        embedding_ax.set_ylim(-1, 1)
        embedding_ax.set(adjustable='box-forced', aspect='equal')

        embedding_ax.set_xlabel(embedding_type)

        embedding_vals = all_embeddings[embedding_type]

        embs_0 = embedding_vals[ckpt_0]
        x0, y0 = np.split(embs_0, 2, axis=1)

        sc = embedding_ax.scatter(x0, y0, color="purple")

        annotations = [embedding_ax.annotate(i, xy=(x, y), fontsize=20)
                       for i, (x, y) in enumerate(zip(x0, y0))]

        for annotation in annotations:
            annotation.set_animated(True)

        return sc, annotations

    sc_annotations = {k: get_sc_annotations(k) for k in ax_embedding_map}

    fig.tight_layout()

    def get_update(i, embedding_type):
        ckpt = checkpoints[i]
        embedding_vals = all_embeddings[embedding_type]
        embs = embedding_vals[ckpt]
        sc, annotations = sc_annotations[embedding_type]
        sc.set_offsets(embs)

        for i, (x, y) in enumerate(embs):
            annotations[i].set_position((x, y))

        checkpoint_label.set_text('Checkpoint: {}'.format(ckpt))

        return [sc] + annotations

    def update(i):
        return flatten([get_update(i, k) for k in ax_embedding_map])

    tf.logging.info("Defining animation function.")
    anim = FuncAnimation(fig, update, frames=len(checkpoints), interval=1,
                         blit=True)

    gif_dir = os.path.join(estimator.model_dir, 'gifs')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    gif_path = os.path.join(gif_dir, 'training.gif')

    tf.logging.info("Building and saving gif.")
    anim.save(gif_path, dpi=80, writer='imagemagick')

    tf.logging.info("Gif saved to {}.".format(gif_path))

    return anim
