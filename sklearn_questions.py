"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples correctly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.

Detailed instructions for question 2:
The data to split should contain the index or one column in
datetime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict on the following. For example if you have data distributed from
November 2020 to March 2021, you have 4 splits. The first split
will allow to learn on November data and predict on December data,
the second split to learn December and predict on January etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, check_array
from sklearn.calibration import check_classification_targets
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """Implement KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """Initialize the classifier."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the KNearestNeighbors model using X and y."""
        # Validate input arrays
        X, y = validate_data(self, X, y)
        # Check classification targets
        check_classification_targets(y)
        # Save unique classes and training data
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels for samples in X."""
        # Check that fit has been called
        check_is_fitted(self, ["X_train_", "y_train_"])
        # Validate input without resetting fitted attributes
        X = validate_data(self, X, reset=False)
        # Prepare array to store predictions
        y_pred = np.empty(X.shape[0], dtype=self.y_train_.dtype)

        # Compute distances between test points and training points
        distances = pairwise_distances(X, self.X_train_, metric="euclidean")
        # Find indices of nearest neighbors
        nearest_index = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        # Extract labels of nearest neighbors
        nearest_labels = self.y_train_[nearest_index]

        # Majority vote among nearest neighbors
        for i, labels in enumerate(nearest_labels):
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred[i] = unique_labels[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """Compute accuracy of the model on X and y."""
        # Check that fit has been called
        check_is_fitted(self, ["X_train_", "y_train_"])
        # Ensure arrays are in correct shape
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        # Predict and compute mean accuracy
        y_predict = self.predict(X)
        return float(np.mean(y_predict == y))


class MonthlySplit(BaseCrossValidator):
    """Implement cross-validator based on monthly splits."""

    def __init__(self, time_col="index"):
        """Initialize the cross-validator."""
        self.time_col = time_col

    def _extract_time(self, X):
        """Extract datetime Series from X.

        Handles both index or a specific column.
        Raises ValueError if column is missing or not datetime.
        """
        # If using index as datetime
        if self.time_col == "index":
            extracted_time = pd.Series(
                X.index, name="extracted_time"
            ).reset_index(drop=True)
        else:
            if self.time_col not in X.columns:
                raise ValueError("Error: time_col not found in input")
            # Extract datetime column
            extracted_time = X[self.time_col].reset_index(drop=True)

        # Ensure column is datetime
        if not pd.api.types.is_datetime64_any_dtype(extracted_time):
            raise ValueError(f"{self.time_col} must be a datetime")

        return pd.to_datetime(extracted_time)

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of monthly splits in X."""
        extracted_time = self._extract_time(X)
        # Convert to periods and count unique months
        distinct_months = extracted_time.dt.to_period("M").unique()
        return max(0, len(distinct_months) - 1)

    def split(self, X, y=None, groups=None):
        """Yield train and test indices for each monthly split."""
        extracted_time = self._extract_time(X)
        # Convert to monthly periods
        months = extracted_time.dt.to_period("M")
        unique_months = np.sort(months.unique())

        # Yield indices for each successive pair of months
        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            # Get train and test indices
            idx_train = np.where(months == train_month)[0]
            idx_test = np.where(months == test_month)[0]

            # Yield training and test indices
            yield idx_train, idx_test
