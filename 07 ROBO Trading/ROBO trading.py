#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import os
from os.path import join
import pandas as pd
import numpy as np
import pickle as pk
import statsmodels.api as sm
import scipy.optimize as sco
from sklearn.linear_model import LinearRegression as LR

#2007_S1
path = os.getcwd()
src = join(path , 'source')
out = join(path , 'output')

TWII = pd.read_csv(join(src , 'TWII.csv') , engine = 'python' , index_col = 0)
TWII.index = pd.to_datetime(TWII.index)

TW150 = pk.load(open(join(src , 'close.dat') , 'rb'))

stock = TW150.dropna(axis = 1)

N = int(len(stock)/13)

log_stock = np.log(stock/stock.shift(N)).dropna(how = 'all')[2:]
log_TWII = np.log(TWII/TWII.shift(1)).dropna(how = 'all')

def reg_lr(a , m):
    reg = LR().fit(a , m)
    return reg.intercept_[0] , reg.coef_[0][0]

# get alpha and beta
def df_reg(df_year , df_tw):
    df = {'alpha':[] , 'beta':[]}
    for col in df_year:
        a = np.array(df_year[col])
        a = a.reshape(len(df_year[col]) , 1)
        A , B = reg_lr(a , df_tw)
        df['alpha'].append(A)
        df['beta'].append(B)
    df = pd.DataFrame(df , index = df_year.columns)
    return df

# 計算報酬率及標準差函數
def Portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights )
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) 
    return std, returns

# 建立風險函數
def Portfolio_volatility(weights, mean_returns, cov_matrix):
    return Portfolio_performance(weights, mean_returns, cov_matrix)[0]

# 找出投資組合中最小風險
def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0,1)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(Portfolio_volatility, num_assets*[1/num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

years = list(np.unique(log_stock.index.year))

value = {'year': years , 'Portfolio' : [100] , 'TWII' : [100]}
#i = 0
for i in range(11):
    year = i + 2007
    
    data = log_stock[log_stock.index.year == year]
    twii = log_TWII[log_TWII.index.year == year]
    
    '''
    指標:
        E(R)/(Std(R)+Skew(R)) - alpha
        越大越好
    '''
    
    # 選指標最大的前五檔股票
    df = df_reg(data , twii)
    T =  ((data.mean()/(data.std() + data.skew())) - df['alpha']).sort_values()
    T = list(T[:5].index)
    
    # 資料
    tarprice = TW150[TW150.index.year == year+1][T]
    tarreturn = log_stock[log_stock.index.year == year+1][T]
    
    # Markowitz Optimization
    mean_return = tarreturn.mean()
    cov_matrix = tarreturn.cov()
    
    w = min_variance(mean_return , cov_matrix)["x"]
    
    # calculate value
    buy_price = tarprice.iloc[0] * 1000
    sell_price = tarprice.iloc[-1] * 1000
    
    units = (value['Portfolio'][i] * w) / buy_price
    spread = sell_price - buy_price
    value['Portfolio'].append( value['Portfolio'][i] + (spread * units).sum())
    
    # 計算大盤報酬
    V_TWII = TWII[TWII.index.year == year+1]
    TWII_buy = V_TWII.iloc[0]
    TWII_sell = V_TWII.iloc[-1]
    
    U_TWII = float(value['TWII'][i] / TWII_buy)
    Spread_TWII = float(TWII_sell - TWII_buy)
    
    value['TWII'].append(value['TWII'][i] + Spread_TWII * U_TWII)

value = pd.DataFrame(value)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot( 'year', 'Portfolio', data=value, color='#F5B041', linewidth=2)
plt.plot( 'year', 'TWII', data=value, color='#5DADE2', linewidth=2)
plt.legend()
plt.grid(True)
plt.title('Portfolio vs TWII')
plt.xlabel('Year')
plt.ylabel('Value')
plt.savefig(join(out ,'Portfolio vs TWII.png'))

R = value.pct_change().dropna()
IRR_P = np.power(value['Portfolio'][len(years)-1]/value['Portfolio'][0] , 1/len(years))-1
IRR_TW = np.power(value['TWII'][len(years)-1]/value['TWII'][0] , 1/len(years))-1

value = value.append({'year':'E(R)' , 'Portfolio':R['Portfolio'].mean() , 'TWII':R['TWII'].mean()} , ignore_index=True)
value = value.append({'year':'IRR' , 'Portfolio':IRR_P , 'TWII':IRR_TW} , ignore_index=True)
value = value.append({'year':'Sigma' , 'Portfolio':R['Portfolio'].std() , 'TWII':R['TWII'].std()} , ignore_index=True)
value = value.append({'year':'Sharpe' , 'Portfolio':IRR_P/R['Portfolio'].std() , 'TWII':IRR_TW/R['TWII'].std()} , ignore_index=True)

value.to_csv(join(out , 'Portfolio vs TWII.csv') , index = False)

'''
==========
Regression
==========
'''
codes = pk.load(open(join(src , 'codes.dat'),'rb'))
writer = pd.ExcelWriter(join(out , 'Regression.xlsx'))

for i in range(len(codes)):
    code = str(codes[i])
    Data = pk.load(open(join(src+'/TALIB_code/dat' , code+'.dat'),'rb')).iloc[:,3:].dropna()
    n = int(len(Data)/len(years))
    
    df = {}
    
    for j in range(len(years)-1):
        
        data = Data[n*j:n*(j+1)]
        
        year = str(years[j])
        Y = data['Rt'][1:].shift(-1).dropna()
        
        # 差分
        x = data.drop(['Rt','MACDsignal','MACDhist'] , axis = 1)
        x = x[1:-1] - x.shift(1)[1:-1]
        
        X = sm.add_constant(x)
        
        est = sm.OLS(Y , X)
        est = est.fit()
        
        params = est.params
        pval = est.pvalues
        
        df[year] = params
        df[year+'_pval'] = pval
    
    df = pd.DataFrame(df)
    df.to_excel(writer , 'TW'+code , index_label = 'params' , float_format = '%.4f')

writer.close()

