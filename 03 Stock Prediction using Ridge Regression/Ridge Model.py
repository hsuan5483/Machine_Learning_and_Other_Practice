#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import numpy as np 
import pandas as pd
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import plotly
from plotly import tools
import plotly.plotly as py
import plotly.graph_objs as go


'''
========
蒐集資料
========
'''

ticker = '^GSPC'
start = '2010-01-01'
end = '2018-12-31'
df = web.data.DataReader(ticker, 'yahoo', start, end)
df = df.drop(['Adj Close'], axis=1)

# defining labels and features for regression
# Label
shift = -10
target = 'Close'
features = ['Open', 'High', 'Low', 'Close', 'Volume']

target_next = target+'_next'+str(-shift)
target_next_pred = target_next+'_pred'
df[target_next] = df[target].shift(shift)
df = df.dropna()


# Features
lag = 30
for i in range(1,lag+1):
    for feature in features:
        df[feature+"_lag"+str(i)] = df[feature].shift(i)
    
df = df.dropna()

'''
=============
分割特徵及目標
=============
'''

X = df.drop([target_next], axis=1)
y = df[target_next]

'''
==========================
為簡化模型不做多項式特徵生成
==========================
'''

'''
=================
分成訓練及測試樣本
=================
'''
#test_size = 0.2
#X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = test_size , random_state = 37)

train_size = 0.8
N = X.shape[0]
Num_train = int(N*train_size)

X_train, y_train = X[:Num_train], y[:Num_train]
X_test, y_test = X[Num_train:], y[Num_train:]


'''
========
建立流程
========
'''
# a
pipeline1 = Pipeline([('scaler', MinMaxScaler()),
                     ('regression', Ridge(random_state = 123457))
                     ])


parameters1 = {'scaler' : [StandardScaler(), MinMaxScaler()],
              'regression__alpha': np.logspace(-3,-1,3),
              }

scoring = 'neg_mean_absolute_error'

n_splits = 5
cv = KFold(n_splits=n_splits, shuffle = True, random_state = 123457)

(model_name, model) = pipeline1.steps[1]
SearchCV_a = GridSearchCV(estimator = pipeline1,
                        param_grid = parameters1,
                        scoring = scoring, 
                        cv = cv,
                        return_train_score = True,
                        verbose = 1, 
                        n_jobs = -1)

SearchCV_a.fit(X_train, y_train)

results_a = pd.DataFrame.from_dict(SearchCV_a.cv_results_)


'''
=====
Score
=====

score = |'mean_test_score'/mean('mean_test_score')-'mean_train_score'/mean('mean_train_score')| + ('std_test_score' + 'std_train_score')
(越小越好)

'''

results_a['score'] = abs((results_a['mean_test_score']/results_a['mean_test_score'].mean() - \
         results_a['mean_train_score']/results_a['mean_train_score'].mean())) + \
        (results_a['std_test_score'] + results_a['std_train_score']) #* (results_a['rank_test_score']*0.01)

best_estimator1 = SearchCV_a.best_estimator_
best_params1 = SearchCV_a.best_params_

best_model1 = best_estimator1.fit(X_train, y_train)
y_test_pred1 = best_model1.predict(X_test)
y_train_pred1 = best_model1.predict(X_train)


best = results_a[results_a['score'] == min(results_a['score'])]

pipeline1_2 = Pipeline([('scaler' , best['param_scaler'][0]),
                       ('regression' , Ridge(alpha = best['param_regression__alpha'][0]))
                       ])

best_model1_2 = pipeline1_2.fit(X_train, y_train)
y_test_pred1_2 = best_model1.predict(X_test)
y_train_pred1_2 = best_model1.predict(X_train)

y_test_pred = pd.concat([y_test , pd.Series(y_test_pred1_2 , index = y_test.index , name = 'y_test_pred1')] , axis = 1)
y_train_pred = pd.concat([y_train , pd.Series(y_train_pred1_2 , index = y_train.index , name = 'y_train_pred1')] , axis = 1)



# b

pipeline2 = Pipeline([('scaler', MinMaxScaler()),
                     ('MLP' , MLPRegressor(solver = 'lbfgs', random_state = 123457))
                     ])
    
parameters2 = {'scaler' : [StandardScaler(), MinMaxScaler()],
              'MLP__alpha': np.logspace(-3,-1,3),
              'MLP__hidden_layer_sizes':[(32) , (64) , (128) , (64,64)],
              }

SearchCV_b = GridSearchCV(estimator = pipeline2,
                        param_grid = parameters2,
                        scoring = scoring, 
                        cv = cv,
                        return_train_score = True,
                        verbose = 1, 
                        n_jobs = -1)

SearchCV_b.fit(X_train, y_train)

results_b = pd.DataFrame.from_dict(SearchCV_b.cv_results_)

best_estimator2 = SearchCV_b.best_estimator_
best_params2 = SearchCV_b.best_params_


best_model2 = best_estimator2.fit(X_train, y_train)
y_test_pred2 = best_model2.predict(X_test)
y_train_pred2 = best_model2.predict(X_train)


results_b['score'] = abs((results_b['mean_test_score']/results_b['mean_test_score'].mean() - \
         results_b['mean_train_score']/results_b['mean_train_score'].mean())) + \
        (results_b['std_test_score'] + results_b['std_train_score']) #* (results_a['rank_test_score']*0.01)

bestb = results_b[results_b['score'] == min(results_b['score'])]
best = best.append(bestb, ignore_index=True)

pipeline2_2 = Pipeline([('scaler', best['param_scaler'][1]),
                     ('MLP' , MLPRegressor(alpha = best['param_MLP__alpha'][1] , hidden_layer_sizes = best['param_MLP__hidden_layer_sizes'][1], solver = 'lbfgs', random_state = 123457))
                     ])

best_model2_2 = pipeline2_2.fit(X_train, y_train)
y_test_pred2_2 = best_model2_2.predict(X_test)
y_train_pred2_2 = best_model2_2.predict(X_train)


y_test_pred['y_test_pred2'] = y_test_pred2_2
y_train_pred['y_train_pred2'] = y_train_pred2_2


# c
#choice = [Ridge(random_state = 123457) , MLPRegressor(solver = 'lbfgs', random_state = 123457)]
#
#pipeline2 = Pipeline([('scaler', MinMaxScaler()),
#                     ('regression' , choice)
#                     ])
#    
#parameters2 = {'scaler' : [StandardScaler(), MinMaxScaler()],
#              'MLP__alpha': np.logspace(-3,-1,3),
#              'MLP__hidden_layer_sizes':[(32) , (64) , (128) , (64,64)],
#              }
#
#SearchCV_b = GridSearchCV(estimator = pipeline2,
#                        param_grid = parameters2,
#                        scoring = scoring, 
#                        cv = cv,
#                        return_train_score = True,
#                        verbose = 1, 
#                        n_jobs = -1)
#
#SearchCV_b.fit(X_train, y_train)

all_results = results_a.append(results_b, ignore_index = True)
#pd.concat([results_a , results_b])
#all_results.index = [x for x in range(len(all_results))]

bestc = all_results[all_results['score'] == min(all_results['score'])]
best = best.append(bestc, ignore_index = True)

# 最好的和(b)一樣


# d

#plot train
train = go.Scatter(
        x = y_train_pred['Close_next10'].index,
        y = y_train_pred['Close_next10'],
        mode = 'lines',
        name = 'train_a'
        )

train_pred_a = go.Scatter(
        x = y_train_pred['Close_next10'].index,
        y = y_train_pred['y_train_pred1'],
        mode = 'lines',
        name = 'train_pred_a'
        )

train_pred_bc = go.Scatter(
        x = y_train_pred['Close_next10'].index,
        y = y_train_pred['y_train_pred2'],
        mode = 'lines',
        name = 'train_pred_b/c'
        )


data1 = [train , train_pred_a , train_pred_bc]

layout1 = dict(title = 'Train'
             )

fig1 = dict(data = data1 , layout = layout1)
plotly.offline.plot(fig1 , filename = 'Train Data.html')

# plot test
test = go.Scatter(
        x = y_test_pred['Close_next10'].index,
        y = y_test_pred['Close_next10'],
        mode = 'lines',
        name = 'test'
        )

test_pred_a = go.Scatter(
        x = y_test_pred['Close_next10'].index,
        y = y_test_pred['y_test_pred1'],
        mode = 'lines',
        name = 'test_pred_a'
        )

test_pred_bc = go.Scatter(
        x = y_test_pred['Close_next10'].index,
        y = y_test_pred['y_test_pred2'],
        mode = 'lines',
        name = 'test_pred_b/c'
        )


data2 = [test , test_pred_a , test_pred_bc]

layout2 = dict(title = 'Test'
             )

fig2 = dict(data = data2 , layout = layout2)
plotly.offline.plot(fig2 , filename = 'Test Data.html')

from sklearn.metrics import mean_squared_error

test_MSE_a = mean_squared_error(y_test_pred['Close_next10'], y_test_pred['y_test_pred1'])
test_MSE_bc = mean_squared_error(y_test_pred['Close_next10'], y_test_pred['y_test_pred2'])
