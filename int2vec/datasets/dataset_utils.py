import numpy as np
import tensorflow as tf


def _split_current_previous_next(a):
    return {'current': a[1:-1], 'previous': a[0:-2], 'next': a[2:]}


def _combine_dicts_by_key(dicts, agg_fn=lambda x: x):
    keys = dicts[0].keys()

    if not all(d.keys() == keys for d in dicts[0:]):
        raise KeyError("Dicts {} do not all share the same keys.".format(dicts))

    return {k: agg_fn([d[k] for d in dicts]) for k in keys}


def combine_chapters_curr_prev_next(chapters):
    chapters_c_p_n = [_split_current_previous_next(chp) for chp in chapters]

    return _combine_dicts_by_key(
        chapters_c_p_n,
        agg_fn=lambda x: np.concatenate(x, axis=0))


def get_input_fn_from_data(data, feature_cols, batch_size,
                           label_cols=None, epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(data)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size)

        if epochs is not None:
            dataset = dataset.repeat(count=epochs)

        iterator = dataset.make_one_shot_iterator()

        next_elements = iterator.get_next()

        next_features = {col: next_elements[col] for col in feature_cols}

        next_labels = None if label_cols is None else {
            col: next_elements[col] for col in label_cols}

        return next_features, next_labels

    return input_fn
