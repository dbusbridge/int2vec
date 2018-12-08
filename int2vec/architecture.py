import tensorflow as tf


_ARCHITECTURE_FEATURES_LABELS = {
    'autoencoder': (['current'], ['current']),
    'skipgram': (['current'], ['previous', 'next'])}


def get_feature_label_cols(architecture):
    if architecture not in _ARCHITECTURE_FEATURES_LABELS:
        raise ValueError("Unknown architecture {}. Must be one of {}".format(
            architecture, tuple(_ARCHITECTURE_FEATURES_LABELS.keys())))

    return _ARCHITECTURE_FEATURES_LABELS[architecture]


def get_architecture_fn(label_cols):
    def architecture_fn(features, params):
        source_embedding = tf.keras.layers.Embedding(
            input_dim=params.n_classes,
            output_dim=params.embed_dim,
            name='integer_embed')

        projection_modules = {
            col: tf.keras.layers.Dense(units=params.n_classes,
                                       name='projection_{}'.format(col)) for
            col in label_cols}

        embeddings = source_embedding(features['current'])

        logits = {col: m(embeddings) for col, m in projection_modules.items()}

        return logits

    return architecture_fn
