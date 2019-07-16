import pickle
import time
from typing import List
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample


def prepare_data(file_path: str, sample_size: int = None) -> (List[str], List[str], List[str], List[str]):
    start = time.time()
    print('Reading training data...')
    df = pd.read_csv(file_path)
    end = time.time()
    print(f'Finished reading training data in %.3f seconds' % (end - start))
    print(df.shape)
    if sample_size is not None:
        df = df.sample(sample_size)

    # missing value impute
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    missing_columns = list(mis_val_table_ren_columns.index[mis_val_table_ren_columns['% of Total Values'] > 60])
    df = df.drop(columns=missing_columns)
    df = df.fillna(df.median())
    df = df.fillna('unknown')

    y = []  # the labels
    data = []  # the features
    target_col = 'TARGET'
    features = list([x for x in df.columns if x != target_col])

    for row in tqdm(df.to_dict('records')):
        y.append(row[target_col])
        data.append({k: row[k] for k in features})

    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2, stratify=y)

    # fix imbalance dataset (upsample y)

    y_train_pd = pd.DataFrame(y_train)  # increased sample size
    X_train_pd = pd.DataFrame(data_train)  # increased sample size
    y_train_pd.columns = ['TARGET']
    Xy = pd.concat([X_train_pd, y_train_pd], axis=1)
    # separate minority and majority classes
    approved = Xy[Xy.TARGET == 0]
    reject = Xy[Xy.TARGET == 1]
    # upsample minority
    reject_upsampled = resample(reject,
                                replace=True,  # sample with replacement
                                n_samples=(len(approved)) // 2,  # match number in majority class
                                random_state=27) # reproducible results
    # combine majority and upsampled minority
    upsampled = pd.concat([approved, reject_upsampled])
    print(f'Upsampled Target Distribution (training data): {upsampled.TARGET.value_counts()}')
    y_train = upsampled.TARGET
    data_train = upsampled.drop('TARGET', axis=1)
    data_train = data_train.to_dict('records')

    return data_train, data_test, y_train, y_test

#don't need?
def prepare_cross_validation_data(X, y, n_folds: int = 5):
    kfold = StratifiedKFold(n_splits=n_folds)

    fold_data = []
    for train_idx, test_idx in kfold.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        fold_data.append((X_train, X_test, y_train, y_test))

    return fold_data


def evaluate_prediction(predictions: 'np.array', y_test: 'np.array'):
    print(classification_report(predictions, y_test))
    accuracy = accuracy_score(predictions, y_test)
    print(f'accuracy: {round(accuracy, 2)}')
    return accuracy


def save_binary(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def load_binary(path):
    return pickle.load(open(path, 'rb'))

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        # print(results)
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


