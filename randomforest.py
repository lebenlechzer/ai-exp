#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn import cross_validation, preprocessing
from sklearn.metrics import explained_variance_score, median_absolute_error
# from sklearn import tree, linear_model
from sklearn.ensemble import RandomForestRegressor

import cPickle as pickle

import pandas as pd


def main():
    # leave out county and state name
    cols = list(range(2, 26))
    data2013 = pd.read_csv("wheat-2013-supervised.csv", usecols=cols)
    data2014 = pd.read_csv("wheat-2014-supervised.csv", usecols=cols)
    data = pd.concat([data2013, data2014])

    # remove columns that have missing values or are not used
    X = data.drop(['Date', 'pressure', 'visibility', 'Yield'], axis=1)
    X = X.drop(['precipIntensity', 'precipIntensityMax',
                'precipProbability'], axis=1)
    X = X.values
    y = data['Yield'].values

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2, random_state=0)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Number of data points: {}".format(len(X_train_scaled)))

    # reg = tree.DecisionTreeRegressor()
    # reg = linear_model.LinearRegression()
    reg = RandomForestRegressor(n_estimators=15)
    scores = cross_validation.cross_val_score(reg, X_train_scaled, y_train,
                                              cv=5, n_jobs=-1, scoring='r2')
    print("Accuracy: {:.3f} (+/- {:.3f})".format(scores.mean(),
                                                 scores.std() * 2))

    # fit on training data (80%), predict test data (20%)
    reg.fit(X_train_scaled, y_train)
    y_true, y_pred = y_test, reg.predict(X_test_scaled)
    print("Explained variance score: {:.2f}"
          .format(explained_variance_score(y_true, y_pred)))
    print("Median absolute error: {:.2f}"
          .format(median_absolute_error(y_true, y_pred)))
    print("Examples:")
    print("True: {}".format(y_true[:10]))
    print("Pred: {}".format(y_pred[:10]))
    pickle.dump(reg, open("regressor.pkl", "wb"))


if __name__ == "__main__":
    main()
