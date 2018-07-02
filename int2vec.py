import os

import numpy as np
import tensorflow as tf

from int2vec.architecture import get_architecture_fn, get_feature_label_cols
from int2vec.datasets import dataset_utils
from int2vec.datasets import numbers
from int2vec.estimator import get_estimator_fn, get_model_fn
from int2vec.figures import make_plots, save_plots

tf.logging.set_verbosity(tf.logging.DEBUG)


tf.flags.DEFINE_integer(name="seed", default=123, help="The random seed.")
tf.flags.DEFINE_integer(name="chapter_size", default=10 ** 3,
                        help="The size of each chapter.")
tf.flags.DEFINE_integer(name="embed_dim", default=2,
                        help="The size of the embedding.")
tf.flags.DEFINE_float(name="learning_rate", default=1.0e-2,
                      help="The learning rate.")
tf.flags.DEFINE_integer(name="epochs", default=100,
                        help="The number of epochs in the training set.")
tf.flags.DEFINE_integer(name="max_steps", default=1000,
                        help="The number of training steps.")
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

FLAGS = tf.flags.FLAGS


def get_train_eval_spec(params):
    train_input_fn, eval_input_fn = dataset_utils.get_train_eval_input_fns(
        dataset=params.dataset,
        feature_cols=params.feature_cols,
        label_cols=params.label_cols,
        chapter_size=params.chapter_size,
        epochs=params.epochs)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=params.max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, steps=100)

    return train_spec, eval_spec


def get_estimator(run_config, params):
    architecture_fn = get_architecture_fn(label_cols=params.label_cols)

    model_fn = get_model_fn(architecture_fn=architecture_fn)

    estimator_fn = get_estimator_fn(model_fn=model_fn)

    return estimator_fn(run_config=run_config, params=params)


def get_run_config_params():
    feature_cols, label_cols = get_feature_label_cols(
        architecture=FLAGS.architecture)

    model_dir = os.path.join(FLAGS.run_dir, FLAGS.dataset, FLAGS.architecture)

    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        tf_random_seed=FLAGS.seed)

    params = tf.contrib.training.HParams(embed_dim=FLAGS.embed_dim,
                                         learning_rate=FLAGS.learning_rate,
                                         n_classes=numbers.NUMBER_CLASSES,
                                         max_steps=FLAGS.max_steps,
                                         feature_cols=feature_cols,
                                         label_cols=label_cols,
                                         chapter_size=FLAGS.chapter_size,
                                         epochs=FLAGS.epochs,
                                         dataset=FLAGS.dataset)

    return run_config, params


def train_estimator(run_config, params):
    estimator = get_estimator(run_config, params)

    train_spec, eval_spec = get_train_eval_spec(params)

    tf.estimator.train_and_evaluate(estimator=estimator,
                                    train_spec=train_spec, eval_spec=eval_spec)

    return estimator


def main(unused_argv):
    run_config, params = get_run_config_params()

    np.random.seed(run_config.tf_random_seed)

    estimator = train_estimator(run_config=run_config, params=params)

    plots = make_plots(params=params, estimator=estimator)

    if FLAGS.save_plots:
        save_plots(plots=plots, run_config=run_config)

    tf.logging.info("Finished!")


if __name__ == "__main__":
    tf.app.run(main=main)
