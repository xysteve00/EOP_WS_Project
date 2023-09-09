"""
    Script for electrisity production prediction 
    with linear regression model(L2 regularization) on wind speed data (train/validate/test).
    author: Xiaoyu Zhang
    date: 09/08/2023
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


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

    ## Build feature engineering pipeline
    steps = [
        ('scalar', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Ridge(alpha=10, fit_intercept=True))
    ]

    pipeline = Pipeline(steps)

    ## Model training
    pipeline.fit(x, y)
    print(f'Training accu: {pipeline.score(x, y)}')
    print(f'Test accu: {pipeline.score(x_val, y_val)}')

    ## Evaluation
    y_hat = pipeline.predict(x_val)
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
    fig.savefig("figures/reg_l2.png")
    # output MAE
    print(f'validation data MAE: {np.mean(abs(y_hat - y_val)):.3f}')
    print(f'validation data MAE from sklearn mae metrics: {mae:.3f}')

if __name__ == "__main__":
    main()

