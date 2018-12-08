import os

import numpy as np
import tensorflow as tf

from int2vec.architecture import get_architecture_fn, get_feature_label_cols
from int2vec.datasets import dataset_utils
from int2vec.datasets import numbers
from int2vec.estimator import get_estimator_fn, get_model_fn
from int2vec.figures import make_plots, save_plots
from int2vec.gifs import make_gif

tf.logging.set_verbosity(tf.logging.DEBUG)


# Important behavioural configurations
tf.flags.DEFINE_string(name="run_dir", default="/tmp/int2vec",
                       help="The location to save and restore results.")
tf.flags.DEFINE_string(name="architecture", default="autoencoder",
                       help="The model architecture. Most be one of "
                            "`autoencoder` or `skipgram`.")
tf.flags.DEFINE_string(name="dataset", default="odd_even",
                       help="The dataset to use. Most be one of "
                            "`odd_even` or `increasing`.")
tf.flags.DEFINE_boolean(name="save_plots", default=True,
                        help="`True` to save plots to model dir, `False` "
                             "otherwise.")
tf.flags.DEFINE_boolean(name="train", default=True,
                        help="`True` to train a model, `False` otherwise.")
tf.flags.DEFINE_boolean(name="make_gif", default=False,
                        help="`True` to make a gif, `False` otherwise.")

# Other standard hyperparameters that you might want to play with
tf.flags.DEFINE_float(name="learning_rate", default=1.0e-2,
                      help="The learning rate.")
tf.flags.DEFINE_integer(name="seed", default=123, help="The random seed.")
tf.flags.DEFINE_integer(name="chapter_size", default=10 ** 3,
                        help="The size of each chapter.")
tf.flags.DEFINE_integer(name="embed_dim", default=2,
                        help="The size of the embedding.")
tf.flags.DEFINE_integer(name="max_steps", default=1000,
                        help="The number of training steps.")
tf.flags.DEFINE_integer(name="save_checkpoints_steps", default=100,
                        help="Checkpoint frequency in steps.")
tf.flags.DEFINE_integer(name="keep_checkpoints_max", default=5,
                        help="Number of checkpoints to keep.")


FLAGS = tf.flags.FLAGS


def get_estimator(run_config, params):
    architecture_fn = get_architecture_fn(label_cols=params.label_cols)

    model_fn = get_model_fn(architecture_fn=architecture_fn)

    estimator_fn = get_estimator_fn(model_fn=model_fn)

    return estimator_fn(run_config=run_config, params=params)


def get_run_config_params():
    feature_cols, label_cols = get_feature_label_cols(
        architecture=FLAGS.architecture)

    model_dir = os.path.join(
        FLAGS.run_dir, FLAGS.dataset, FLAGS.architecture, str(FLAGS.embed_dim))

    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        tf_random_seed=FLAGS.seed,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoints_max)

    tf.logging.info("Run config: {}".format(run_config.__dict__))

    params = tf.contrib.training.HParams(
        embed_dim=FLAGS.embed_dim,
        learning_rate=FLAGS.learning_rate,
        n_classes=numbers.NUMBER_CLASSES,
        max_steps=FLAGS.max_steps,
        feature_cols=feature_cols,
        label_cols=label_cols,
        chapter_size=FLAGS.chapter_size,
        dataset=FLAGS.dataset,
        architecture=FLAGS.architecture)

    tf.logging.info("Hyparameters: {}".format(params))

    return run_config, params


def main(unused_argv):
    if FLAGS.make_gif and FLAGS.embed_dim > 2:
        raise ValueError(
            "Cannot produce gifs for embedding dimension greater than 2.")

    run_config, params = get_run_config_params()

    np.random.seed(run_config.tf_random_seed)

    train_input_fn = dataset_utils.get_train_input_fn(
        dataset=params.dataset,
        feature_cols=params.feature_cols,
        label_cols=params.label_cols,
        chapter_size=params.chapter_size)

    estimator = get_estimator(run_config, params)

    if FLAGS.train:
        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.max_steps)

    if FLAGS.save_plots:
        plots = make_plots(params=params, estimator=estimator)
        save_plots(plots=plots, run_config=run_config)

    if FLAGS.make_gif:
        make_gif(params=params, estimator=estimator)

    tf.logging.info("Finished!")


if __name__ == "__main__":
    tf.app.run(main=main)
