import numpy as np
import tensorflow as tf

from sklearn.preprocessing import normalize

from int2vec.architecture import get_architecture_fn
from int2vec.datasets import odd_even, dataset_utils
from int2vec.estimator import get_estimator_fn, get_model_fn
from int2vec.figures import plot_embeddings

tf.logging.set_verbosity(tf.logging.DEBUG)
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer(name='seed', default=123, help='The random seed.')
tf.app.flags.DEFINE_integer(name='chapter_size', default=10 ** 3,
                            help='The size of each chapter.')
tf.app.flags.DEFINE_integer(name='embed_dim', default=2,
                            help='The size of the embedding.')
tf.app.flags.DEFINE_float(name='learning_rate', default=1.0e-2,
                          help='The learning rate.')
tf.app.flags.DEFINE_integer(name='epochs', default=100,
                            help='The number of epochs in the training set.')
tf.app.flags.DEFINE_integer(name='max_steps', default=1000,
                            help='The number of training steps.')
tf.app.flags.DEFINE_string(name='model_dir', default='/tmp/int2vec/autoencoder',
                           help='The location to save and restore results.')

np.random.seed(FLAGS.seed)

feature_cols, label_cols = ['current'], ['current']

train_data = odd_even.get_data(size=FLAGS.chapter_size)
eval_data = odd_even.get_data(size=FLAGS.chapter_size)

train_input_fn = dataset_utils.get_input_fn_from_data(
    data=train_data,
    feature_cols=feature_cols, label_cols=label_cols,
    batch_size=FLAGS.chapter_size, epochs=FLAGS.epochs)

eval_input_fn = dataset_utils.get_input_fn_from_data(
    data=eval_data,
    feature_cols=feature_cols, label_cols=label_cols,
    batch_size=FLAGS.chapter_size, epochs=1)

params = tf.contrib.training.HParams(
    embed_dim=FLAGS.embed_dim,
    learning_rate=FLAGS.learning_rate,
    n_classes=odd_even.NUMBER_CLASSES,
    max_steps=FLAGS.max_steps)

run_config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                    tf_random_seed=FLAGS.seed)

architecture_fn = get_architecture_fn(label_cols=label_cols)

model_fn = get_model_fn(architecture_fn=architecture_fn)

estimator_fn = get_estimator_fn(model_fn=model_fn)

estimator = estimator_fn(run_config=run_config, params=params)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                    max_steps=params.max_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=train_input_fn, steps=100)

tf.estimator.train_and_evaluate(estimator=estimator,
                                train_spec=train_spec, eval_spec=eval_spec)

embeddings = estimator.get_variable_value(name='integer_embed/embeddings')
embeddings = normalize(embeddings, norm='l2', axis=1)

plot_embeddings(embeddings, title='int2vec autoencoder integer embeddings')
