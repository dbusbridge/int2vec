import tensorflow as tf

from tensorflow.python.estimator.model_fn import ModeKeys


def _get_train_op_fn(params):
    optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
    global_step = tf.train.get_global_step()

    def train_op_fn(loss):
        return optimizer.minimize(loss=loss, global_step=global_step)

    return train_op_fn


def get_model_fn(architecture):
    def model_fn(features, labels, mode, params):
        """Model function used in the estimator.

        Args:
            features (dict): Input features to the model.
            labels (dict): Labels for training and evaluation.
            mode (ModeKeys): Specifies if training, evaluation or inferring..
            params (HParams): The Hyperparameters.

        Returns:
            (EstimatorSpec) The model to be run by Estimator.

        """
        logits = architecture(features, params)

        # If mode is inference, return the predictions
        if mode == ModeKeys.PREDICT:
            class_probabilities = tf.nn.softmax(logits,
                                                name='class_probabilities')
            predictions = {
                'logits': logits,
                'classes': tf.argmax(input=logits, axis=1),
                'class_probabilities': class_probabilities}

            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions)

        head = tf.contrib.estimator.multi_class_head(n_classes=params.n_classes)

        return head.create_estimator_spec(
            features=features,
            mode=mode,
            logits=logits,
            labels=labels,
            train_op_fn=_get_train_op_fn(params=params))

    return model_fn


def get_estimator_fn(model_fn):

    def estimator_fn(run_config, params):
        """Return the model as a TensorFlow Estimator object.

        Args:
            run_config (RunConfig): Configuration for Estimator run.
            params (HParams): The hyperparameters.

        Returns: Estimator.

        """
        return tf.estimator.Estimator(
            model_fn=model_fn,
            params=params,
            config=run_config)

    return estimator_fn
