"""
    Script for electrisity production prediction 
    with XGBoost model on wind speed data (train/validate/test).
    author: Xiaoyu Zhang
    date: 09/08/2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    ## Data Analysis
    df = pd.read_excel('data/Wind_data.xlsx', sheet_name=[0, 1])
    df_train, df_test = df.get(0), df.get(1)


    data = df_train
    data.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(15, 10))
    sns_plot = sns.heatmap(data.corr(), cmap="PiYG")
    ax.set_title('Pearson Correlation', fontdict={'fontsize':18, 'fontweight':'bold'})
    fig = sns_plot.get_figure()
    fig.savefig("figures/covar.png")

    ## Data Processing and data split
    # shuffle data samples
    df_train = df_train.sample(frac = 1)

    # extract hour, day, month feature
    df_train['hour'] = df_train['DATETIME'].dt.hour
    df_train['month'] = df_train['DATETIME'].dt.month
    df_train['day'] = df_train['DATETIME'].dt.day

    # split dataset
    Y = df_train[['CF']]
    # X = df_train.set_index('DATETIME')
    X = df_train.drop(['DATETIME','CF'], axis=1)
    # X = df_train.drop(['CF'], axis=1)
    x_train, x_val,y_train, y_val = X[:-92], X[-92:], Y[:-92], Y[-92:]
    x, y, x_val,y_val = x_train.values, y_train.values, x_val.values, y_val.values

    ## XGBoostRegressor
    xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)

    ## Engineer new features to improve the results
    sc_x = StandardScaler()

    # Apply scale and normalize training data
    X_new = sc_x.fit_transform(x)

    # model fitting
    xgb_model.fit(X_new, y)

    # prediction
    X_val_new = sc_x.transform(x_val)
    y_hat = xgb_model.predict(X_val_new)
    y_hat = y_hat.reshape(y_hat.shape[0],-1)
    print(f'Training accu: {xgb_model.score(X_new, y)}')
    print(f'Test accu: {xgb_model.score(X_val_new, y_val)}')
    mae = mean_absolute_error(y_val, y_hat)

    ## Visualize predication 
    results = np.concatenate((y_hat, y_val), axis=1)
    df_res = pd.DataFrame(results, columns=["Pred", "Val"])           
    ## Set figure size
    fig, ax = plt.subplots(figsize = (10,6))
    # Plot the curve
    sns_plot = sns.lineplot(data=df_res)
    ax.set_title('CF_WS Prediction on Test Data', fontdict={'fontsize':14, 'fontweight':'bold'})
    # Set label for x-axis
    ax.set_xlabel( "Time" , fontdict= { 'fontsize': 13, 'fontweight':'bold'})
    # Set label for y-axis
    ax.set_ylabel( "CF" , fontdict= { 'fontsize': 13, 'fontweight':'bold'})
    fig = sns_plot.get_figure()
    fig.savefig("figures/xgb_val.png")
    # output MAE
    print(f'XGBoost validation data MAE: {np.mean(abs(y_hat - y_val)):.3f}')
    print(f'XGBoost validation data MAE from sklearn mae metrics: {mae:.3f}')

    ## Process Test Data
    # extract hour, day, month feature
    df_test['hour'] = df_test['DATETIME'].dt.hour
    df_test['month'] = df_test['DATETIME'].dt.month
    df_test['day'] = df_test['DATETIME'].dt.day


    # X = df_train.set_index('DATETIME')
    x_test = df_test.drop(['DATETIME','CF'], axis=1)
    x_test = x_test.values

    ## All data samples
    # split dataset
    Y = df_train[['CF']]
    X = df_train.drop(['DATETIME','CF'], axis=1)
    x_all, y_all = X.values, Y.values

    ## XGBoostRegressor
    xgb_model_all = xgb.XGBRegressor(objective="reg:linear", random_state=42)

    ## engineer new features to improve the results

    sc_x_all = StandardScaler()

    # scale and normalize training data
    X_new_all = sc_x_all.fit_transform(x_all)

    # model fitting
    xgb_model_all.fit(X_new_all, y_all)

    # prediction
    X_val_new_all = sc_x_all.transform(x_test)
    y_hat_all = xgb_model_all.predict(X_val_new_all)
    y_hat_test_all = y_hat_all.reshape(y_hat_all.shape[0],-1)

    # prediction
    X_test_new = sc_x.transform(x_test)
    y_hat_test = xgb_model.predict(X_test_new)
    y_hat_test = y_hat_test.reshape(y_hat_test.shape[0],-1)

    #visualize predication 
    ## Prepare dataframe
    results = np.concatenate((y_hat_test, y_hat_test_all), axis=1)
    df_res = pd.DataFrame(results, columns=["trained by x_train data", "Trained by all data"]) 
    ## Set figure size
    fig, ax = plt.subplots(figsize = (8,4))
    ## Plot the curve
    sns_plot = sns.lineplot(data=df_res)
    ax.set_title('CF_WS Prediction on Test Data', fontdict={'fontsize':16, 'fontweight':'bold'})
    ## Set label for x-axis
    ax.set_xlabel( "Time" , fontdict= { 'fontsize': 14, 'fontweight':'bold'})
    ## Set label for y-axis
    ax.set_ylabel( "CF" , fontdict= { 'fontsize': 14, 'fontweight':'bold'})
    print(f'final prediction:\n{y_hat_test_all}')
    fig = sns_plot.get_figure()
    fig.savefig("figures/xgb_test.png")
if __name__ == "__main__":
    main()

