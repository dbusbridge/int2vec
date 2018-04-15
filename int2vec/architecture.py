import sonnet as snt


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
        integer_embed = snt.Embed(
            vocab_size=params.n_classes, embed_dim=params.embed_dim,
            name='integer_embed')

        projection_modules = {
            col: snt.Linear(output_size=params.n_classes,
                            name='projection_{}'.format(col)) for
            col in label_cols}

        embeddings = integer_embed(features['current'])

        logits = {col: m(embeddings) for col, m in projection_modules.items()}

        return logits

    return architecture_fn
