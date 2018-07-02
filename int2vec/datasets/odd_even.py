import numpy as np

from int2vec.datasets import dataset_utils
from int2vec.datasets import numbers


def _get_even_chapter(size):
    return np.random.choice(numbers.EVEN_NUMBERS, size=(size,), replace=True)


def _get_odd_chapter(size):
    return np.random.choice(numbers.ODD_NUMBERS, size=(size,), replace=True)


def get_data(size):
    chapters = _get_even_chapter(size), _get_odd_chapter(size)

    return dataset_utils.combine_chapters_curr_prev_next(chapters)
