
"""Tensorflow losses"""

import tensorflow as tf

MSE_NAME = "MSE"
CC_NAME = "CC"
R2_NAME = "R2"
NEGATIVE_CC_NAME = "negCC"
NEGATIVE_R2_NAME = "negR2"
POISSON_LOG_LIKELIHOOD_NAME = "PoissonLL"
CATEGORICAL_CROSS_ENTROPY_NAME = "CCE"
SPARSE_CATEGORICAL_CROSS_ENTROPY_NAME = "SCCE"


def masked_mse(mask_value=None):
    """Returns a tf MSE loss computation function, but with support for setting one value as a mask indicator

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(y_true, y_pred):
        # mse = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true), axis=-1) # Without handling NaNs
        # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
        sh = tf.shape(y_true)
        y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
        y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
        y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
        y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)
        if mask_value is not None:
            mask_value_cast = tf.constant(mask_value, dtype=y_true_f.dtype)
            isOk = tf.not_equal(y_true_f, mask_value_cast)
        else:
            isOk = tf.ones_like(y_true_f, dtype=bool)
        isOk1 = tf.math.reduce_all(isOk, axis=-1)
        y_true_masked = tf.boolean_mask(y_true_f, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(y_pred_f, isOk1, axis=0)
        lossFunc = tf.keras.losses.MeanSquaredError()
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = MSE_NAME
    return f


def compute_CC(x, y):  # https://stackoverflow.com/a/58890795/2275605
    """Computes correlation coefficient (CC) in tensorflow

    Args:
        x (numpy array): input 1
        y (numpy array): input 2

    Returns:
        tf.Tensor: CC value
    """
    mx = tf.math.reduce_mean(x)
    my = tf.math.reduce_mean(y)
    xm, ym = x - mx, y - my
    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)
    return r_num / r_den


def _undefined_r2(dtype):
    """Return a scalar undefined R2 value with the requested dtype."""
    return tf.cast(float("nan"), dtype)


def _finite_mean(values):
    """Average finite values and return NaN only when none are available."""
    finite = tf.boolean_mask(values, tf.math.is_finite(values))
    return tf.cond(
        tf.size(finite) > 0,
        lambda: tf.reduce_mean(finite),
        lambda: _undefined_r2(values.dtype),
    )


def compute_R2(y_true, y_pred):  # https://stackoverflow.com/a/58890795/2275605
    """Compute per-dimension R2 with NaN for undefined flat targets.

    Parameters
    ----------
    y_true, y_pred : tf.Tensor, shape (samples, dimensions)
        Matched target and prediction arrays.

    Returns
    -------
    tf.Tensor, shape (dimensions,)
        Per-dimension R2 values. Dimensions with zero target variance are NaN.
    """
    dtype = y_pred.dtype
    samples = tf.cast(tf.shape(y_true)[0], dtype)
    total = tf.math.reduce_sum(y_true, axis=0)
    m_true = tf.math.divide_no_nan(total, samples)
    r_num = tf.math.reduce_sum(
        tf.math.squared_difference(y_true, y_pred),
        axis=0,
    )
    r_den = tf.math.reduce_sum(
        tf.math.squared_difference(y_true, m_true),
        axis=0,
    )
    valid = tf.logical_and(
        samples > 0,
        tf.logical_and(
            r_den > 0,
            tf.logical_and(
                tf.math.is_finite(r_num),
                tf.math.is_finite(r_den),
            ),
        ),
    )
    values = 1 - tf.math.divide_no_nan(r_num, r_den)
    undefined = tf.fill(tf.shape(values), _undefined_r2(dtype))
    return tf.where(valid, values, undefined)


def computeCC_masked(y_true, y_pred, mask_value=None):
    """Computes correlation coefficient (CC) in tensorflow, with support for a masked value.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the CC computation.
    Args:
        y_true (numpy array): input 1.
        y_pred (numpy array): input 2
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.

    Returns:
        tf.Tensor: CC value
    """
    # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
    sh = tf.shape(y_true)
    y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
    y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)
    if mask_value is not None:
        mask_value_cast = tf.constant(mask_value, dtype=y_true_f.dtype)
        isOk = tf.not_equal(y_true_f, mask_value_cast)
    else:
        isOk = tf.ones_like(y_true_f, dtype=bool)
    isOk1 = tf.math.reduce_all(isOk, axis=-1)
    y_true_masked = tf.boolean_mask(y_true_f, isOk1, axis=0)
    y_pred_masked = tf.boolean_mask(y_pred_f, isOk1, axis=0)
    CC = compute_CC(y_true_masked, y_pred_masked)
    return CC


def computeR2_masked(y_true, y_pred, mask_value=None):
    """Computes correlation of determination (R2) in tensorflow, with support for a masked value.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the CC computation.
    Args:
        y_true (numpy array): input 1.
        y_pred (numpy array): input 2
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.

    Returns:
        tf.Tensor: R2 value
    """
    # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
    sh = tf.shape(y_true)
    y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
    y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)
    if mask_value is not None:
        mask_value_cast = tf.constant(mask_value, dtype=y_true_f.dtype)
        isOk = tf.not_equal(y_true_f, mask_value_cast)
    else:
        isOk = tf.ones_like(y_true_f, dtype=bool)
    isOk1 = tf.math.reduce_all(isOk, axis=-1)
    y_true_masked = tf.boolean_mask(y_true_f, isOk1, axis=0)
    y_pred_masked = tf.boolean_mask(y_pred_f, isOk1, axis=0)
    R2 = compute_R2(y_true_masked, y_pred_masked)
    return R2


def masked_CC(mask_value=None):
    """Returns a tf correlation coefficient (CC) computation function, but with support for setting one value as a mask indicator.
    Takes mean of CC across dimensions. See computeCC_masked for details of computing CC for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        meanCC = tf.math.reduce_mean(
            computeCC_masked(y_true, y_pred, mask_value)
        )  # Average across dimensions
        return meanCC

    f.__name__ = CC_NAME
    return f


def masked_R2(mask_value=None):
    """Return a batch R2 function that excludes undefined dimensions.

    Parameters
    ----------
    mask_value : number, optional
        Marker that excludes an observation row; default None.
    """

    def f(y_true, y_pred):
        return _finite_mean(computeR2_masked(y_true, y_pred, mask_value))

    f.__name__ = R2_NAME
    return f


class MaskedR2(tf.keras.metrics.Metric):
    """Aggregate finite per-dimension R2 values across one Keras epoch.

    Parameters
    ----------
    mask_value : number, optional
        Marker that excludes an observation row; default None.
    name : str, optional
        Keras history key; default "R2".
    dtype : str or tf.dtypes.DType, optional
        Metric accumulator dtype; default TensorFlow's configured dtype.
    """

    def __init__(self, mask_value=None, name=R2_NAME, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.mask_value = mask_value
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulate finite R2 values without counting flat dimensions."""
        if sample_weight is not None:
            raise ValueError("MaskedR2 does not support sample weights.")
        values = computeR2_masked(y_true, y_pred, self.mask_value)
        valid = tf.math.is_finite(values)
        finite = tf.where(valid, values, tf.zeros_like(values))
        self.total.assign_add(tf.reduce_sum(tf.cast(finite, self.dtype)))
        self.count.assign_add(
            tf.reduce_sum(tf.cast(valid, self.dtype))
        )

    def result(self):
        """Return NaN only when the complete epoch had no valid R2 values."""
        return tf.cond(
            self.count > 0,
            lambda: tf.math.divide_no_nan(self.total, self.count),
            lambda: _undefined_r2(self.dtype),
        )

    def reset_state(self):
        """Clear the epoch accumulators."""
        self.total.assign(0)
        self.count.assign(0)

    def get_config(self):
        """Return serializable construction parameters."""
        config = super().get_config()
        config.update({"mask_value": self.mask_value})
        return config


def masked_negativeCC(mask_value=None):
    """Returns a tf negative correlation coefficient (CC) computation function, but with support for setting one value as a mask indicator.
    Takes mean of negative CC across dimensions. See computeCC_masked for details of computing CC for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        meanCC = tf.math.reduce_mean(
            computeCC_masked(y_true, y_pred, mask_value)
        )  # Average across dimensions
        return -meanCC

    f.__name__ = NEGATIVE_CC_NAME
    return f


def masked_negativeR2(mask_value=None):
    """Returns a tf negative correlation of determination (R2) computation function, but with support for setting one value as a mask indicator.
    Takes mean of negative R2 across dimensions. See computeR2_masked for details of computing R2 for each dimension.
    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to
    """

    def f(y_true, y_pred):
        return -_finite_mean(computeR2_masked(y_true, y_pred, mask_value))

    f.__name__ = NEGATIVE_R2_NAME
    return f


def masked_PoissonLL_loss(mask_value=None):
    """Returns a tf function that computes the poisson negative log likelihood loss, with support for setting one value as a mask indicator.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the loss computation.

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(true_counts, pred_logLambda):
        sh = tf.shape(true_counts)
        true_counts_f = tf.reshape(true_counts, [tf.reduce_prod(sh[:-1]), sh[-1]])
        pred_logLambda_f = tf.reshape(pred_logLambda, [tf.reduce_prod(sh[:-1]), sh[-1]])
        if mask_value is not None:
            mask_value_cast = tf.constant(int(mask_value), dtype=true_counts_f.dtype)
            isOk = tf.not_equal(true_counts_f, mask_value_cast)
        else:
            isOk = tf.ones_like(true_counts_f, dtype=bool)
        isOk1 = tf.math.reduce_all(isOk, axis=-1)
        y_true_masked = tf.boolean_mask(true_counts_f, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(pred_logLambda_f, isOk1, axis=0)
        # LL = true_counts_f * pred_logLambda_f - tf.math.exp(pred_logLambda_f) - tf.math.lgamma( true_counts_f+1 )
        # pLoss = - tf.reduce_mean(tf.boolean_mask(LL, isOk))
        # https://www.tensorflow.org/api_docs/python/tf/keras/losses/poisson
        lossFunc = tf.keras.losses.Poisson()
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = POISSON_LOG_LIKELIHOOD_NAME
    return f


def masked_CategoricalCrossentropy(mask_value=None):
    """Returns a tf function that computes the Categorical Crossentropy loss, but with support for setting one value as a mask indicator.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the loss computation.

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(y_true, y_pred):
        # Assumes that the last two dimensions are the only data dimensions, others are sample dimensions (batch,time,etc)
        sh = tf.shape(y_true)
        y_true = tf.reshape(y_true, [tf.reduce_prod(sh[:-2]), sh[-2], sh[-1]])
        y_pred = tf.reshape(y_pred, [tf.reduce_prod(sh[:-2]), sh[-2], sh[-1]])
        if mask_value is not None:
            mask_value_cast = tf.constant(int(mask_value), dtype=y_true.dtype)
            isOk = tf.not_equal(y_true, mask_value_cast)
        else:
            isOk = tf.ones_like(y_true, dtype=bool)
        isOk1 = tf.math.reduce_all(
            isOk, axis=tf.range(tf.rank(isOk) - 2, tf.rank(isOk))
        )
        y_true_masked = tf.boolean_mask(y_true, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(y_pred, isOk1, axis=0)
        lossFunc = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )  # Later will need softmax for pred_model
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = CATEGORICAL_CROSS_ENTROPY_NAME
    return f


def masked_SparseCategoricalCrossentropy(mask_value=None):
    """Returns a tf function that computes the Sparse Categorical Crossentropy loss, but with support for setting one value as a mask indicator.
    First dimension of data is the sample/time dimension. If a sample has a mask_value in
    one of its dimensions, it will be discarded before the loss computation.

    Args:
        mask_value (numpy value, optional): if not None, will treat this value as mask indicator. Defaults to None.
    """

    def f(y_true, y_pred):
        # Assumes that the last dimension is the only data dimension, others are sample dimensions (batch,time,etc)
        sh = tf.shape(y_true)
        y_true = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
        sh2 = tf.shape(y_pred)
        y_pred = tf.reshape(y_pred, [tf.reduce_prod(sh2[:-2]), sh2[-2], sh2[-1]])
        if mask_value is not None:
            mask_value_cast = tf.constant(int(mask_value), dtype=y_true.dtype)
            isOk = tf.not_equal(y_true, mask_value_cast)
        else:
            isOk = tf.ones_like(y_true, dtype=bool)
        isOk1 = tf.math.reduce_all(isOk, axis=-1)
        y_true_masked = tf.boolean_mask(y_true, isOk1, axis=0)
        y_pred_masked = tf.boolean_mask(y_pred, isOk1, axis=0)
        lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )  # Later will need softmax for pred_model
        return lossFunc(y_true_masked, y_pred_masked)

    f.__name__ = SPARSE_CATEGORICAL_CROSS_ENTROPY_NAME
    return f
