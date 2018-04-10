import numpy as np
import sonnet as snt
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

from int2vec.corpora import autoencoder_data, get_input_fn, NUMBER_CLASSES
from int2vec.estimator import get_estimator_fn, get_model_fn

plt.style.use('seaborn-white')


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

np.random.seed(FLAGS.seed)

data = autoencoder_data(size=FLAGS.chapter_size)
input_fn = get_input_fn(data=data,
                        batch_size=FLAGS.chapter_size, epochs=FLAGS.epochs)

params = tf.contrib.training.HParams(
    embed_dim=FLAGS.embed_dim,
    learning_rate=FLAGS.learning_rate,
    n_classes=NUMBER_CLASSES,
    max_steps=FLAGS.max_steps)

run_config = tf.estimator.RunConfig(model_dir='/tmp/int2vec/autoencoder',
                                    tf_random_seed=FLAGS.seed)


def architecture(features, params):
    integer_embed = snt.Embed(
        vocab_size=params.n_classes, embed_dim=params.embed_dim,
        name='integer_embed')
    projection = snt.Linear(output_size=params.n_classes)

    net = integer_embed(features['x'])
    net = projection(net)

    return net


model_fn = get_model_fn(architecture=architecture)
estimator_fn = get_estimator_fn(model_fn=model_fn)

estimator = estimator_fn(run_config=run_config, params=params)

train_spec = tf.estimator.TrainSpec(input_fn=input_fn,
                                    max_steps=params.max_steps)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=1)

tf.estimator.train_and_evaluate(estimator=estimator,
                                train_spec=train_spec, eval_spec=eval_spec)

embeddings = estimator.get_variable_value(name='integer_embed/embeddings')
embeddings = normalize(embeddings, norm='l2', axis=1)

# Plot the embedding (don't worry - this is supposed to suck!)
fig, ax = plt.subplots()
for i, (x, y) in enumerate(embeddings):
    ax.scatter(x, y, color='purple'), ax.annotate(i, xy=(x, y), fontsize=20)
ax.set_title('Autoencoder int2vec', fontsize=30)
# fig.savefig('img/int2vec_auto.png',
#             dpi=300, figsize=(10, 10), bbox_inches='tight')
