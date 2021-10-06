import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import datasets
from sklearn import model_selection


def create_folds(ds, label_name, n_folds=5):
    """

    :param label_name: the column name of labels (y)
    :param n_folds: number of folds
    :param ds: the dataset for k-fold cross validation
    :type ds: DataFrame
    """
    if not isinstance(ds, DataFrame):
        return 'Should be a DataFrame'
    ds['kfold'] = -1
    ds = ds.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Sturge's rule
    num_bins = int(np.floor(1 + np.log2(len(ds))))

    # bin targets
    ds.loc[:, 'bins'] = pd.cut(
        ds[label_name], bins=num_bins, labels=False
    )
    # Stratified K-fold
    # kf = model_selection.StratifiedKFold(n_splits=n_folds)
    # Random K-fold
    kf = model_selection.KFold(n_splits=n_folds, random_state=None, shuffle=True)

    for f, (t_, v_) in enumerate(kf.split(X=ds, y=ds.bins.values)):
        ds.loc[v_, 'kfold'] = f

    ds = ds.drop('bins', axis=1)
    return ds


if __name__ == '__main__':
    df_mnist = pd.read_csv('../input/mnist_train.csv')
    df_mnist_kfold = create_folds(df_mnist, 'label')
    df_mnist_kfold.to_csv('../input/mnist_train_folds.csv')
