import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import process_data
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import copy

def gbrt(output_index):
    """
    gradient boosted regression tree
    print MSE value and deviance plot

    Args:
        output_index: int, Stage1 output feature 의 인덱스(0~14)
    """
    data_set = process_data.Data()

    output_actual = data_set.primary_output_actual.iloc[:, output_index]
    output_set = data_set.primary_output_setpoint.iloc[:, output_index]

    y = output_actual
    x = data_set.data_all.iloc[:, 1:42]
    # print(x_train.shape)
    x = pd.concat([x, output_set], axis = 1)
    # print(x_train.shape)

    # 파라미터
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=13)

    params = {'n_estimators': 550,
            'max_depth': 5,
            'min_samples_split': 5,
            'learning_rate': 0.01,
            'loss': 'ls'}

    # gbrt 적용
    reg = ensemble.GradientBoostingRegressor(**params)
    reg.fit(x_train, y_train)
    # x_plot = np.linspace (0, 14000, num=14000, retstep=True)
    # plt.plot(x, reg.predict(x[:, np.newaxis]), linewidth=2)

    mse = mean_squared_error(y_test, reg.predict(x_test))
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # 시각화
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    for i, y_pred in enumerate(reg.staged_predict(x_test)):
        test_score[i] = reg.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    plt.title('Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
            label='Training Set Deviance')
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
            label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    fig.tight_layout()
    plt.show()

def random_forest(output_index):
    """
    random forest

    Args:
        output_index: int, Stage1 output feature 의 인덱스(0~14)
    """
    data_set = process_data.Data()

    output_actual = data_set.primary_output_actual.iloc[:, output_index]
    output_set = data_set.primary_output_setpoint.iloc[:, output_index]

    y = output_actual
    x = data_set.data_all.iloc[:, 1:42]
    x = pd.concat([x, output_set], axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.3, random_state = 13
    )
    
    params = {
        'n_estimators': 10, # The number of trees in the forest.
        'random_state': 0,
        'max_depth': 8,
        'n_jobs': -1,
        'min_samples_split': 6
    }

    
    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)
    print("Train score for property%d: " %output_index, model.score(x_train, y_train))
    print("Test score for property%d: " %output_index, model.score(x_test, y_test))

    feature_list = list(x.columns)
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    x_values = list(range(len(importances)))
    plt.style.use('fivethirtyeight')
    plt.bar(x_values, importances, orientation = 'vertical')
    plt.xticks(x_values, feature_list, rotation  = 'vertical')

    # plt.barh(x_values, importances, align = 'center')
    # plt.yticks(x_values, importances)
    plt.ylabel('Importance')
    plt.xlabel('Variable')
    plt.title('Variable Importances')
    plt.show()
    
    y_predict = model.predict(x_test)
    plt.scatter(np.arange(len(x_test)), y_predict, c="darkorange")
    plt.scatter(np.arange(len(x_test)), y_test, c="green")
    plt.show()

    plt.plot(model.predict(x), "b", linewidth=1)
    plt.plot(y, "r", linewidth=1)
    plt.show()

for i in range(15):
    random_forest(i)





