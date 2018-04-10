import tensorflow as tf
import numpy as np

# Set of evens from 0 to 8 and odds from 1 to 9 inclusive
EVEN_NUMBERS, ODD_NUMBERS = list(range(0, 10, 2)), list(range(1, 10, 2))
NUMBERS = EVEN_NUMBERS + ODD_NUMBERS
NUMBER_CLASSES = len(NUMBERS)


def _get_even_chapter(size):
    return np.random.choice(
        a=EVEN_NUMBERS, size=(size,), replace=True)


def _get_odd_chapter(size):
    return np.random.choice(
        a=ODD_NUMBERS, size=(size,), replace=True)


def autoencoder_data(size):
    odd_chapter, even_chapter = _get_even_chapter(size), _get_odd_chapter(size)

    x = np.concatenate([even_chapter, odd_chapter], axis=0)

    return {'current': x}


def _build_skipgram(a):
    current, previous, next = a[1:-1], a[0:-2], a[2:]

    return {'current': current, 'previous': previous, 'next': next}


def skipgram_data(size):
    odd_chapter, even_chapter = _get_even_chapter(size), _get_odd_chapter(size)

    odd_skipgrams = _build_skipgram(odd_chapter)
    even_skipgrams = _build_skipgram(even_chapter)

    current = np.concatenate([odd_skipgrams['current'],
                              even_skipgrams['current']], axis=0)
    previous = np.concatenate([odd_skipgrams['previous'],
                              even_skipgrams['previous']], axis=0)
    next = np.concatenate([odd_skipgrams['next'],
                           even_skipgrams['next']], axis=0)

    return {'current': current, 'previous': previous, 'next': next}


def get_input_fn(data, feature_label_fn, batch_size, epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size)

        if epochs is not None:
            dataset = dataset.repeat(count=epochs)

        iterator = dataset.make_one_shot_iterator()
        next_elements = iterator.get_next()

        return feature_label_fn(next_elements)

    return input_fn
