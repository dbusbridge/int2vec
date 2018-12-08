import numpy as np
import tensorflow as tf

from int2vec.datasets import circle
from int2vec.datasets import odd_even

DATASETS = {"odd_even": odd_even, "circle": circle}


def _split_current_previous_next(a):
    return {"current": a[1:-1], "previous": a[0:-2], "next": a[2:]}


def _combine_dicts_by_key(dicts, agg_fn=lambda x: x):
    keys = dicts[0].keys()

    if not all(d.keys() == keys for d in dicts[0:]):
        raise KeyError("Dicts {} do not all share the same keys.".format(dicts))

    return {k: agg_fn([d[k] for d in dicts]) for k in keys}


def combine_chapters_curr_prev_next(chapters):
    chapters_c_p_n = [_split_current_previous_next(chp) for chp in chapters]

    return _combine_dicts_by_key(chapters_c_p_n,
                                 agg_fn=lambda x: np.concatenate(x, axis=0))


def get_input_fn_from_data(data, feature_cols, batch_size,
                           label_cols=None, epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.repeat(count=epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        next_elements = iterator.get_next()
        next_features = {col: next_elements[col] for col in feature_cols}

        next_labels = None if label_cols is None else {
            col: next_elements[col] for col in label_cols}

        return next_features, next_labels

    return input_fn


def get_train_input_fn(
    dataset, feature_cols, label_cols, chapter_size, epochs=None):
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset {}. Must be one of {}".format(
            dataset, tuple(DATASETS.keys())))

    data = DATASETS[dataset]

    train_data = data.get_data(size=chapter_size)

    train_input_fn = get_input_fn_from_data(
        data=train_data,
        feature_cols=feature_cols, label_cols=label_cols,
        batch_size=chapter_size, epochs=epochs)

    return train_input_fn
