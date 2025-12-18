import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, check_array
from sklearn.calibration import check_classification_targets
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """Initialize the classifier."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the KNearestNeighbors classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : KNearestNeighbors
            Fitted classifier.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict the class labels for the given samples.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_test_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = validate_data(self, X, reset=False)
        y_pred = np.empty(X.shape[0], dtype=self.y_train_.dtype)
        distances = pairwise_distances(X, self.X_train_, metric='euclidean')
        nearest_index = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_train_[nearest_index]

        for i, labels in enumerate(nearest_labels):
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred[i] = unique_labels[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to score.
        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy of predictions.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator with monthly splits.

    Each split trains on one month and tests on the following month.
    The time column can be a DataFrame column or the index.
    """

    def __init__(self, time_col='index'):
        """Initialize the cross-validator."""
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splits."""
        if self.time_col == 'index':
            extracted_time = pd.Series(X.index, name='extracted_time')
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"{self.time_col} column not found in input")
            extracted_time = X[self.time_col].reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(extracted_time):
            raise ValueError(f"{self.time_col} must be a datetime")

        extracted_time = pd.to_datetime(extracted_time)
        distinct_months = extracted_time.dt.to_period('M').unique()
        return max(0, len(distinct_months) - 1)

    def split(self, X, y=None, groups=None):
        """Generate train/test indices for monthly splits."""
        if self.time_col == 'index':
            extracted_time = pd.Series(X.index, name='extracted_time')
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"{self.time_col} column not found in input")
            extracted_time = X[self.time_col].reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(extracted_time):
            raise ValueError(f"{self.time_col} must be a datetime")

        extracted_time = pd.to_datetime(extracted_time)
        months = extracted_time.dt.to_period('M')
        unique_months = np.sort(months.unique())

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]
            idx_train = np.where(months == train_month)[0]
            idx_test = np.where(months == test_month)[0]
            yield idx_train, idx_test
