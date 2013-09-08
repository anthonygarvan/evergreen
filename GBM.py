

class CustomGBM:
    def __init__(self):
        self.clf = MultinomialNB()

def fit(self, X, y, sample_weight=None):
    """Build a boosted classifier/regressor from the training set (X, y).

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        The training input samples.

    y : array-like of shape = [n_samples]
        The target values (integers that correspond to classes in
        classification, real numbers in regression).

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights. If None, the sample weights are initialized to
        1 / n_samples.

    Returns
    -------
    self : object
        Returns self.
    """
    # Check parameters
    if self.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")

    # Check data
    X, y = check_arrays(X, y, sparse_format="dense")

    y = column_or_1d(y, warn=True)

    if ((getattr(X, "dtype", None) != DTYPE) or
            (X.ndim != 2) or (not X.flags.contiguous)):
        X = np.ascontiguousarray(array2d(X), dtype=DTYPE)

    if sample_weight is None:
        # Initialize weights to 1 / n_samples
        sample_weight = np.empty(X.shape[0], dtype=np.float)
        sample_weight[:] = 1. / X.shape[0]
    else:
        # Normalize existing weights
        sample_weight = np.copy(sample_weight) / sample_weight.sum()

        # Check that the sample weights sum is positive
        if sample_weight.sum() <= 0:
            raise ValueError(
                "Attempting to fit with a non-positive "
                "weighted number of samples.")

    # Clear any previous fit results
    self.estimators_ = []
    self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float)
    self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float)

    for iboost in xrange(self.n_estimators):
        # Boosting step
        sample_weight, estimator_weight, estimator_error = self._boost(
            iboost,
            X, y,
            sample_weight)

        # Early termination
        if sample_weight is None:
            break

        self.estimator_weights_[iboost] = estimator_weight
        self.estimator_errors_[iboost] = estimator_error

        # Stop if error is zero
        if estimator_error == 0:
            break

        sample_weight_sum = np.sum(sample_weight)

        # Stop if the sum of sample weights has become non-positive
        if sample_weight_sum <= 0:
            break

        if iboost < self.n_estimators - 1:
            # Normalize
            sample_weight /= sample_weight_sum

    return self
