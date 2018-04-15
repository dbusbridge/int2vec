import numpy as np

from int2vec.datasets import dataset_utils

# Set of evens from 0 to 8 and odds from 1 to 9 inclusive
EVEN_NUMBERS, ODD_NUMBERS = list(range(0, 10, 2)), list(range(1, 10, 2))
NUMBERS = EVEN_NUMBERS + ODD_NUMBERS
NUMBER_CLASSES = len(NUMBERS)


def _get_even_chapter(size):
    return np.random.choice(a=EVEN_NUMBERS, size=(size,), replace=True)


def _get_odd_chapter(size):
    return np.random.choice(a=ODD_NUMBERS, size=(size,), replace=True)


def get_data(size):
    chapters = _get_even_chapter(size), _get_odd_chapter(size)

    return dataset_utils.combine_chapters_curr_prev_next(chapters)
