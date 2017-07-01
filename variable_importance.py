import sys
import time
import numpy as np
from progress import ProgressBar


def gen_var_impact_scores(clf, X_train, y_train, X_test,
                              n_rounds=100, verbose=False, fit=True):
    """
    Given classifier and training data, return variable importances on the
    test data on a per-sample basis.
    
    Impact scores represent the ability of a given variable
    to influence the result (given the others are held constant).
    """
    if fit:
        clf.fit(X_train, y_train)
    real_scores = clf.predict_proba(X_test)[:, 1]
    impact_scores = np.zeros(X_test.shape)
    if verbose:
        pc = ProgressCounter()
    for var in range(X_train.shape[1]):
        single_var_scores = np.zeros([X_test.shape[0], n_rounds])
        X_test_mod = np.copy(X_test)
        for j in range(n_rounds):
            if verbose:
                pc.increment_progress()
            X_test_mod[:, var] = np.random.choice(X_train[:, var], X_test.shape[0], replace=True)
            single_var_scores[:, j] = clf.predict_proba(X_test_mod)[:, 1]
        impact_scores[:, var] = np.std(single_var_scores, axis=1)
    return impact_scores


def gen_var_result_scores(clf, X_train, X_test, y_train=None,
                              n_rounds=100, verbose=False, fit=True):
    """
    Given classifier and training data, return variable importances on the
    test data on a per-sample basis. UPDATED.
    
    Result scores represent the difference between the observed score and
    the mean score obtained if the specific variable is resampled from the
    training data randomly.
    """
    if fit:
        if verbose:
            print 'Training model...'
            sys.stdout.flush()
            t0 = time.time()
        clf.fit(X_train, y_train)
        if verbose:
            t1 = time.time()
            print 'Training took %.2f seconds' % (t1 - t0)
    real_scores = clf.predict_proba(X_test)[:, 1]
    result_scores = np.zeros(X_test.shape)
    if verbose:
        pb = ProgressBar()
        progress = 0
    for var in range(X_train.shape[1]):
        single_var_scores = np.zeros([X_test.shape[0], n_rounds])
        X_test_mod = np.copy(X_test)
        for j in range(n_rounds):
            if verbose:
                progress += 1
                pb.update_progress(progress / float(n_rounds * X_train.shape[1]))
            X_test_mod[:, var] = np.random.choice(X_train[:, var], X_test.shape[0], replace=True)
            single_var_scores[:, j] = clf.predict_proba(X_test_mod)[:, 1]
        result_scores[:, var] = np.abs(real_scores - np.mean(single_var_scores, axis=1))
    return result_scores


def _resampled_scores(clf, X_train, X_test, n_rounds=100, verbose=False):
    scores = np.zeros([X_test.shape[0], n_rounds, X_test.shape[1]])
    if verbose:
        pc = ProgressCounter()
    for var in range(X_train.shape[1]):
        single_var_scores = np.zeros([X_test.shape[0], n_rounds])
        X_test_mod = np.copy(X_test)
        for j in range(n_rounds):
            if verbose:
                pc.increment_progress()
            X_test_mod[:, var] = np.random.choice(X_train[:, var], X_test.shape[0], replace=True)
            single_var_scores[:, j] = clf.predict_proba(X_test_mod)[:, 1]
        scores[:, :, var] = single_var_scores
    return scores
                      