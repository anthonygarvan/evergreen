__author__ = 'root'
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
from numpy.core.umath_tests import inner1d

class SparseAdaBoost:
    def __init__(self, n_estimators=2, learning_rate=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def _make_estimator(self):
        return BernoulliNB()

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)
        return pred
        #if self.n_classes_ == 2:
        #    return self.classes_.take(pred > 0, axis=0)

        #return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        # The weights are all 1. for SAMME.R
        pred = np.mean([estimator.predict_proba(X)[:,1] for estimator in self.estimators_], axis=0)

        #pred /= self.estimator_weights_.sum()

        #pred[:, 0] *= -1
        return pred

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

            if sample_weight is None:
                # Initialize weights to 1 / n_samples
                #sample_weight = np.empty(X.shape[0], dtype=np.float)
                #sample_weight[:] = 1. / X.shape[0]
                sample_weight = np.ones(X.shape[0], dtype=np.float)
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
                sample_weight, estimator_weight, estimator_error = self._boost_real(
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
            print 'final sample weight: '
            print sample_weight
            return self


    def _boost_real(self, iboost, X, y, sample_weight):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator()

        #try:
        #    estimator.set_params(random_state=self.random_state)
        #except ValueError:
        #    pass

        estimator.fit(X, y, sample_weight=sample_weight)
        self.estimators_.append(estimator)
        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Construct y coding as described in Zhu et al [2]:
        #
        #    y_k = 1 if c == k else -1 / (K - 1)
        #
        # where K == n_classes_ and c, k in [0, K) are indices along the second
        # axis of the y coding with c being the index corresponding to the true
        # class label.
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        y_predict_proba[y_predict_proba <= 0] = 1e-5

        # Boost weight using multi-class AdaBoost SAMME.R alg
        estimator_weight = (-1. * self.learning_rate
                            * (((n_classes - 1.) / n_classes) *
                               inner1d(y_coding, np.log(y_predict_proba))))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)))

        return sample_weight, 1., estimator_error


def _samme_proba(estimator, n_classes, X):
    """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    """
    proba = estimator.predict_proba(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    proba[proba <= 0] = 1e-5
    log_proba = np.log(proba)

    return (n_classes - 1) * (log_proba - (1. / n_classes)
                          * log_proba.sum(axis=1)[:, np.newaxis])
