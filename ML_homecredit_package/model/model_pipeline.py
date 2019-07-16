import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import pprint as pp
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

import ML_homecredit_package.config as config
from ML_homecredit_package.util import preprocess, utils

"""
From `svm.py`, we take one step further to evaluate our SVM classifier on 5-fold cross validation
"""


def get_pipeline(perform_svd: bool = False):
    vectorizer = DictVectorizer()
    scaler = MaxAbsScaler()
    #pca = PCA()
    svd = TruncatedSVD()
    clf = XGBClassifier()
    pipelines = []

    if perform_svd:
        pipelines.extend([
            ('vectorizer', vectorizer),
            ('scaler', scaler),
            ('svd', svd)])

    pipelines.append(('clf', clf))
    #pp.pprint(sorted(clf.get_params().keys()))

    return Pipeline(pipelines)


def run_model(training_data_path: str, perform_svd: bool = False):
    data_train, data_test, y_train, y_test = utils.prepare_data(file_path= training_data_path)
    # above, we can set up sample size to test the codes with small sample size

    pl = get_pipeline(perform_svd=perform_svd)

    search_params = {
        'clf__learning_rate': [0.02, 0.04],
        'clf__min_child_weight': [30, 60],
        'clf__max_depth': [3, 5, 7],
        'clf__subsample': [0.4, 0.2],
        'clf__n_estimators': [100, 200, 300],
        'clf__colsample_bytree': [0.6, 0.4],
        'clf__reg_lambda': [1, 2],
        'clf__reg_alpha': [1, 2],
        'svd__n_components': [10, 20, 50, 100, 150, 200],
    }

    n_iter_search = 20
    cv = RandomizedSearchCV(pl, param_distributions=search_params, n_iter=n_iter_search,
                            scoring={'score': 'balanced_accuracy'}, n_jobs=-1, cv=3,
                            refit='score')

    cv.fit(data_train, y_train)
    print(cv.best_estimator_)

    predictions = cv.predict(data_test)
    acc = utils.evaluate_prediction(predictions=predictions, y_test=y_test)
    print(acc)


if __name__ == '__main__':
    run_model(perform_svd=True, training_data_path=os.path.join(config.DATA_SUBDIR, 'final.csv'))