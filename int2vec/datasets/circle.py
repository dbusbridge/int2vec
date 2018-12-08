import numpy as np

from int2vec.datasets import dataset_utils
from int2vec.datasets import numbers


def _chapter(size, noise_range=1):
    repetitions = np.floor(size / numbers.NUMBER_CLASSES).astype(np.int32)
    repetitions = repetitions + 1

    shift = np.random.randint(low=0, high=numbers.NUMBER_CLASSES)

    sequence_to_repeat = np.array(numbers.ORDERED_NUMBERS)
    sequence_to_repeat = np.roll(sequence_to_repeat, shift=shift)

    repeated_sequences = np.tile(sequence_to_repeat, reps=repetitions)
    repeated_sequences = repeated_sequences[:size]

    sequence_noise = np.random.randint(
        low=-noise_range, high=noise_range, size=repeated_sequences.shape)

    chapter = repeated_sequences + sequence_noise
    chapter = np.remainder(chapter, numbers.NUMBER_CLASSES)

    return chapter


def get_data(size):
    chapters = _chapter(size=size),

    return dataset_utils.combine_chapters_curr_prev_next(chapters)
