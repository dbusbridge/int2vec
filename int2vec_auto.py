import numpy as np
import tensorflow as tf
import tensorlayer as tl

import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

plt.style.use('seaborn-white')


flags = tf.flags
logging = tf.logging

flags.DEFINE_integer(
    flag_name='seed', default_value=123,
    docstring='The random seed.')
flags.DEFINE_integer(
    flag_name='chapter_size', default_value=5 * 10 ** 2,
    docstring='The size of each chapter.')
flags.DEFINE_integer(
    flag_name='embedding_size', default_value=2,
    docstring='The size of the embedding.')
flags.DEFINE_float(
    flag_name='learning_rate', default_value=1e-2,
    docstring='The learning rate for the optimiser.')
flags.DEFINE_integer(
    flag_name='epochs', default_value=2500,
    docstring='The number of epochs to train for.')


FLAGS = flags.FLAGS

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


# Set of even numbers from 0 to 8 inclusive
even_numbers = range(0, 10, 2)

# Set of odd numbers from 1 to 9 inclusive
odd_numbers = range(1, 10, 2)


# Build the corpus
chapter_even_numbers = np.random.choice(
    a=even_numbers, size=FLAGS.chapter_size, replace=True)
chapter_odd_numbers = np.random.choice(
    a=odd_numbers, size=FLAGS.chapter_size, replace=True)

# Autoencoder #################################################################

# Build the data for the number predicting itself
auto_x_even = [np.array(chapter_even_numbers[i])
               for i in range(FLAGS.chapter_size)]

auto_y_even = [np.array(chapter_even_numbers[i])
               for i in range(FLAGS.chapter_size)]

auto_x_odd = [np.array(chapter_odd_numbers[i])
              for i in range(FLAGS.chapter_size)]

auto_y_odd = [np.array(chapter_odd_numbers[i])
              for i in range(FLAGS.chapter_size)]

# Combine the data
auto_x_all = auto_x_even + auto_x_odd
auto_y_all = auto_y_even + auto_y_odd

# Build the model
with tf.variable_scope('autoencoder_int2vec'):
    with tf.variable_scope('inputs'):
        auto_x = tf.placeholder(tf.int32, shape=(None,), name="x")
        auto_y_ = tf.placeholder(tf.int32, shape=(None,), name="y_")

    with tf.variable_scope('embedding'):
        auto_net_emb = tl.layers.EmbeddingInputlayer(
            auto_x,
            vocabulary_size=10,
            embedding_size=FLAGS.embedding_size,
            name='auto_embedding')

    with tf.variable_scope('output'):
        auto_net = tl.layers.DenseLayer(
            auto_net_emb, n_units=10, act=tf.identity, name='auto_output')

    with tf.variable_scope('loss'):
        auto_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=auto_y_, logits=auto_net.outputs)
        auto_loss = tf.reduce_sum(auto_loss, name='auto_loss')

    with tf.variable_scope('optimiser'):
        auto_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        auto_train_op = auto_op.minimize(auto_loss)

# Train it!
auto_sess = tf.Session()
tl.layers.initialize_global_variables(auto_sess)

for e in range(FLAGS.epochs):
    if e % 500 == 0:
        loss = auto_sess.run(
            auto_loss, feed_dict={auto_x: auto_x_all, auto_y_: auto_y_all})
        print('### Epoch: {e}/{max_e} loss: {loss:.2f} ###'.format(
            e=e, max_e=FLAGS.epochs, loss=loss))
    auto_sess.run(
        auto_train_op,
        feed_dict={auto_x: auto_x_all, auto_y_: auto_y_all})


# Get the embeddings for each number
auto_embeddings = auto_sess.run(
    auto_net_emb.outputs, feed_dict={auto_x: np.arange(10)})
# Normalise them
auto_embeddings = normalize(auto_embeddings, norm='l2', axis=1)


# Plot the embedding (don't worry - this is supposed to suck!)
fig, ax = plt.subplots()
for i, (x, y) in enumerate(auto_embeddings):
    ax.scatter(x, y, color='purple')
    ax.annotate(i, xy=(x, y), fontsize=20)
ax.set_title('Autoencoder int2vec', fontsize=30)
# fig.savefig('img/int2vec_auto.png',
#             dpi=300, figsize=(10, 10), bbox_inches='tight')
