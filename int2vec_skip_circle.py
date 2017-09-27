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
    flag_name='vocab_size', default_value=10,
    docstring='The size of the vocab.')
flags.DEFINE_integer(
    flag_name='chapter_size', default_value=1000,
    docstring='The size of the chapter.')
flags.DEFINE_integer(
    flag_name='chapter_noise', default_value=2,
    docstring='The noise across the chapter.')
flags.DEFINE_integer(
    flag_name='embedding_size', default_value=2,
    docstring='The size of the embedding.')
flags.DEFINE_float(
    flag_name='learning_rate', default_value=1e-3,
    docstring='The learning rate for the optimiser.')
flags.DEFINE_integer(
    flag_name='epochs', default_value=2500,
    docstring='The number of epochs to train for.')


FLAGS = flags.FLAGS

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)


# Set of even numbers from 0 to 8 inclusive
increasing_numbers = list(range(0, FLAGS.vocab_size, 1))
min_inc_num, max_inc_num = min(increasing_numbers), max(increasing_numbers)

chapter = increasing_numbers * int(np.ceil(FLAGS.chapter_size /
                                           FLAGS.vocab_size))
chapter = np.array(chapter[:FLAGS.chapter_size])
chapter_noise = np.random.randint(
    low=-FLAGS.chapter_noise,
    high=FLAGS.chapter_noise,
    size=FLAGS.chapter_size)

# Add noise and Fix out of vocabulary numbers
chapter = chapter + chapter_noise
chapter[chapter < min_inc_num] = min_inc_num
chapter[chapter > max_inc_num] = max_inc_num


# Skip-gram ################################################################

# Build the data for the numbers either side of itself
# Build the data for the numbers either side of itself
circular_skip_x = [np.array(chapter[i])
                   for i in range(1, FLAGS.chapter_size - 1)]

circular_skip_y = [np.array([chapter[i-1], chapter[i+1]])
                   for i in range(1, FLAGS.chapter_size - 1)]

# Combine the data
skip_x_all = circular_skip_x
skip_y_all = circular_skip_y

# Build the model
with tf.variable_scope('skipgram_int2vec'):
    with tf.variable_scope('inputs'):
        skip_x = tf.placeholder(tf.int32, shape=(None,), name="x")
        skip_y_ = tf.placeholder(tf.int32, shape=(None, 2), name="y_")

    with tf.variable_scope('embedding'):
        skip_net_emb = tl.layers.EmbeddingInputlayer(
            skip_x,
            vocabulary_size=FLAGS.vocab_size,
            embedding_size=FLAGS.embedding_size,
            name='skip_embedding')

    with tf.variable_scope('output'):
        skip_net = tl.layers.DenseLayer(
            skip_net_emb, n_units=FLAGS.vocab_size,
            act=tf.identity, name='skip_output')

    with tf.variable_scope('loss'):
        skip_y_prev, skip_y_post = tf.unstack(skip_y_, axis=1)

        skip_loss_prev = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=skip_y_prev, logits=skip_net.outputs)
        skip_loss_prev = tf.reduce_sum(skip_loss_prev, name='skip_prev_loss')

        skip_loss_post = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=skip_y_post, logits=skip_net.outputs)
        skip_loss_post = tf.reduce_sum(skip_loss_post, name='skip_post_loss')

        skip_loss = tf.add(skip_loss_prev, skip_loss_post)

    with tf.variable_scope('optimiser'):
        skip_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        skip_train_op = skip_op.minimize(skip_loss)


# Train it!
skip_sess = tf.Session()
tl.layers.initialize_global_variables(skip_sess)

for e in range(FLAGS.epochs):
    if e % 500 == 0:
        loss = skip_sess.run(
            skip_loss, feed_dict={skip_x: skip_x_all, skip_y_: skip_y_all})
        print('### Epoch: {e}/{max_e} loss: {loss:.2f} ###'.format(
            e=e, max_e=FLAGS.epochs, loss=loss))
    skip_sess.run(
        skip_train_op,
        feed_dict={skip_x: skip_x_all, skip_y_: skip_y_all})


# Get the embeddings for each number
skip_embeddings = skip_sess.run(
    skip_net_emb.outputs, feed_dict={skip_x: np.arange(FLAGS.vocab_size)})
# Normalise them
skip_embeddings = normalize(skip_embeddings, norm='l2', axis=1)


# Plot the embedding
fig, ax = plt.subplots()
for i, (x, y) in enumerate(skip_embeddings):
    ax.scatter(x, y, color='purple')
    ax.annotate(i, xy=(x, y), fontsize=20)
ax.set_title('Skipgram circle int2vec', fontsize=30)
# fig.savefig('img/int2vec_skip_circle.png',
#             dpi=300, figsize=(10, 10), bbox_inches='tight')
