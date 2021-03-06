import config
import model_dispatcher
import pathlib
import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree
import argparse


def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    # training data is where kfold is not equal to provided fold
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values

    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)

    preds = clf.predict(x_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f'Fold={fold}, Accuracy={accuracy}')

    joblib.dump(clf,
                pathlib.Path(f'{config.MODEL_OUTPUT}{model}_{fold}.bin'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fold',
        type=int
    )

    parser.add_argument(
        '--model',
        type=str
    )

    # read the arguments form the command line
    args = parser.parse_args()
    run(
        fold=args.fold,
        model=args.model
    )