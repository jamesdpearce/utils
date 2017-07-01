import sys
import time

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.cross_validation import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from ml_utilities import cross_val_predict_proba

class StackedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, final_estimator=LogisticRegression(),
                 include_original_features=False,
                 cv=None,
                 verbose=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.include_original_features = include_original_features
        self.cv = cv
        self.verbose = verbose

    def fit(self, X, y):
        X_meta = np.zeros([X.shape[0], len(self.estimators)])
        if self.cv is None:
            self.cv = KFold(X.shape[0])
        for i, clf in enumerate(self.estimators):
            if self.verbose:
                print 'Generating meta features using model:'
                print str(clf)
            # Generate meta features
            X_meta[:, i] = cross_val_predict_proba(clf, X, y, cv=self.cv, verbose=self.verbose)
            # Train meta-feature generator
            clf.fit(X, y)
        if self.include_original_features:
            X_meta = np.hstack(X, X_meta)
        if self.verbose:
            print 'Fitting final model...'
        self.final_estimator.fit(X_meta, y)

    def predict_proba(self, X):
        X_meta = np.zeros([X.shape[0], len(clf_list)])
        for i, clf in enumerate(self.estimators):
            try:
                X_meta[:, i] = clf.predict_proba(X)[:, 1]
            except:
                X_meta[:, i] = clf.predict(X)
        if include_original_features:
            X_meta = np.hstack(X, X_meta)
        probs = self.clf_final.predict_proba(X_meta)
        return probs
    
    def predict(self, X):
        probs = self.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds

    
class StackedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators, final_estimator=LogisticRegression(),
                 include_original_features=False,
                 cv=None,
                 verbose=0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.include_original_features = include_original_features
        self.cv = cv
        self.verbose = verbose

    def fit(self, X, y):
        X_meta = np.zeros(X.shape[0], len(estimators))
        for i, clf in enumerate(self.estimators):
            # Generate meta features
            X_meta[:, i] = cross_val_predict(clf, X, y, cv=self.cv, verbose=self.verbose)
            # Train meta-feature generator
            clf.fit(X, y)
        if self.include_original_features:
            X_meta = np.hstack(X, X_meta)
        if self.verbose:
            print 'Fitting final model...'
        self.final_estimator.fit(X_meta, y)
    
    def predict(self, X):
        X_meta = np.zeros([X.shape[0], len(clf_list)])
        for i, clf in enumerate(self.estimators):
            X_meta[:, i] = clf.predict(X)
        if include_original_features:
            X_meta = np.hstack(X, X_meta)
        preds = self.clf_final.predict(X_meta)
        return preds