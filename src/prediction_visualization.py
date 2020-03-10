#library
import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import lightgbm as lgb
import time
from sklearn.model_selection import StratifiedKFold
import gc
from tqdm import tqdm_notebook
import plotly.offline as py
py.init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings("ignore")
pd.set_option('max_colwidth', 500)
pd.set_option('max_columns', 500)
pd.set_option('max_rows', 100)
from sklearn.metrics import cohen_kappa_score
def kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def build_model(train, test):
    cols_to_use = ['Type', 'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                   'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                   'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID', 'health', 'Free', 'score',
                   'VideoAmt', 'PhotoAmt', 'AdoptionSpeed', 'No_name', 'Pure_breed', 'desc_length', 'desc_words',
                   'averate_word_length', 'magnitude']

    train = train[[col for col in cols_to_use if col in train.columns]]
    test = test[[col for col in cols_to_use if col in test.columns]]

    cat_cols = ['Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
                'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
                'Sterilized', 'Health', 'State', 'RescuerID',
                'No_name', 'Pure_breed', 'health', 'Free']

    indexer = {}
    for col in cat_cols:
        # print(col)
        _, indexer[col] = pd.factorize(train[col].astype(str))

    for col in tqdm_notebook(cat_cols):
        # print(col)
        train[col] = indexer[col].get_indexer(train[col].astype(str))
        test[col] = indexer[col].get_indexer(test[col].astype(str))

    y = train['AdoptionSpeed']
    train = train.drop(['AdoptionSpeed'], axis=1)

    return train, y, test, cat_cols

def model_train(train, y, test, cat_cols):
    n_fold = 5
    folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=15)
    params = {'num_leaves': 512,
              #  'min_data_in_leaf': 60,
              'objective': 'multiclass',
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 3,
              "bagging_fraction": 0.9,
              "bagging_seed": 11,
              #  "lambda_l1": 0.1,
              # "lambda_l2": 0.1,
              "random_state": 42,
              "verbosity": -1,
              "num_class": 5}

    result_dict = {}
    oof = np.zeros((len(train), 5))
    prediction = np.zeros((len(test), 5))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(train, y)):
        gc.collect()
        print('Fold', fold_n + 1, 'started at', time.ctime())
        X_train, X_valid = train.iloc[train_index], train.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        valid_data = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols)

        model = lgb.train(params,
                          train_data,
                          num_boost_round=20000,
                          valid_sets=[train_data, valid_data],
                          verbose_eval=500,
                          early_stopping_rounds=200)

        del train_data, valid_data
        y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        del X_valid
        gc.collect()
        y_pred = model.predict(test, num_iteration=model.best_iteration)
        oof[valid_index] = y_pred_valid

        scores.append(kappa(y_valid, y_pred_valid.argmax(1)))
        print('Fold kappa:', kappa(y_valid, y_pred_valid.argmax(1)))
        print('')
        prediction += y_pred

        fold_importance = pd.DataFrame()
        fold_importance["feature"] = train.columns
        fold_importance["importance"] = model.feature_importance()
        fold_importance["fold"] = fold_n + 1
        feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    feature_importance["importance"] /= n_fold
    cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
        by="importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
    result_dict['feature_importance'] = feature_importance

    result_dict['prediction'] = prediction
    result_dict['oof'] = oof
    return best_features, oof, model

def plot_importance_of_feature(best_features):
    plt.figure(figsize=(16, 12));
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False),
                palette="OrRd_r");
    plt.title('Importance of Features of five folds');

def plot_confusion_matrix(oof, y):
    prediction_train = oof.argmax(1)
    assert (len(prediction_train) == len(y))
    min_rating = min(min(prediction_train), min(y))
    max_rating = max(max(prediction_train), max(y))
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(prediction_train, y):
        conf_mat[a - min_rating][b - min_rating] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix');

def save_plot_prediction(best_features, oof, y):
    plt.figure(figsize=(16, 12));
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False),
                palette="OrRd_r");
    plt.title('Importance of Features of five folds');
    plt.savefig('output_plots/Importance_of_Features.png')

    prediction_train = oof.argmax(1)
    assert (len(prediction_train) == len(y))
    min_rating = min(min(prediction_train), min(y))
    max_rating = max(max(prediction_train), max(y))
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(prediction_train, y):
        conf_mat[a - min_rating][b - min_rating] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, cmap="YlGnBu", fmt="d", ax=ax)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix');
    plt.savefig('output_plots/Confusion_matrix.png')


def main():
    from data_analysis import read_data, data_prepreocessing
    if not os.path.exists('output_plots'):
        os.mkdir('output_plots')
    train, test, all_data, breeds, colors, states, all_count, sentiment_dict = read_data('data')
    train, test, all_data, breeds, colors = data_prepreocessing(train, test, all_data, breeds, colors, sentiment_dict)
    train, y, test, cat_cols = build_model(train, test)
    best_features, oof, model = model_train(train, y, test, cat_cols)
    save_plot_prediction(best_features, oof, y)

if __name__ == '__main__':
	main()
