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
    y = x

    return {'x': x, 'y': y}


def get_input_fn(data, batch_size, epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size)

        if epochs is not None:
            dataset = dataset.repeat(count=epochs)

        iterator = dataset.make_one_shot_iterator()
        next_elements = iterator.get_next()

        next_features = {'x': next_elements['x']}
        next_labels = {'y': next_elements['y']}

        return next_features, next_labels

    return input_fn
