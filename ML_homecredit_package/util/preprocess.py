import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction import DictVectorizer


#don't need?
def preprocess(images: 'np.array') -> 'np.array':
    """
    Input images is a (n_samples, 8, 8) matrix.
    To apply a classifier on this data, we need to flatten the image, i.e.,
    turn the data in a (samples, n_dim) matrix, where n_dim = 8*8

    :param images: a 3D matrix with shape (n_samples, 8, 8)
    :return: a flattened image matrix of shape (n_samples, 64)
    """

    # raw images, as pixels, are already in matrix format.
    # So a simple reshaping operation (to reshape images from 3D matrix to 2D)
    # is sufficient for this simple dataset.

    print(f'Shape before preprocessing: {images.shape}')
    n_samples = images.shape[0]
    data = images.reshape((n_samples, -1))

    print(f'Shape after preprocessing: {data.shape}')

    return data

#don't need?
def process_data(images_train: 'np.array', images_test: 'np.array') -> (np.array, np.array):
    """
    Process the training and testing raw images into flattened matrices
    :param images_train: the raw images for training
    :param images_test: the raw images for testing
    :return: X_train, X_test
    """

    print(f'Transforming training images...')
    images_train = preprocess(images=images_train)

    print(f'\nTransforming test images...')
    images_test = preprocess(images=images_test)

    return images_train, images_test


def transform_data(X, vectorizer: 'DictVectorizer' = None, scaler: 'MaxAbsScaler' = None, pca: 'PCA' = None):
    if vectorizer is None:
        # this is for training
        vectorizer = DictVectorizer()
        X = X.to_dict('records')
        X = vectorizer.fit_transform(X)
    else:
        # this is for testing
        X = vectorizer.transform(X)

    if scaler is None:
        # this is for training
        scaler = MaxAbsScaler()
        X = scaler.fit_transform(X)
    else:
        # this is for testing
        X = scaler.transform(X)

    if pca is None:
        # this is for training
        pca = PCA(n_components=20)
        X = pca.fit_transform(X)
    else:
        # this is for testing
        X = pca.transform(X)

    return X, scaler, pca
