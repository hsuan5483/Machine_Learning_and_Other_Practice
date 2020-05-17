#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu

Email: amyhsu0619@gmail.com
"""

import os
from os.path import join
import pandas as pd
import numpy as np
import csv
import scipy.optimize as sco
from scipy.optimize import fsolve
from sklearn.linear_model import LinearRegression as LR
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import math
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls

path = os.getcwd()
src = join(path , 'source')
out = join(path , 'output')

# 讀代碼檔轉成list
richinx = list(pd.read_excel(join(src , 'RICHINX.xlsx')).astype(str)['code'] + '.TWO')
data = pd.read_excel(join(src , 'code.xlsx')).astype(str)
bonds = list(data['code'][630:])
data_5g = [x for x in data['5G'] if x != 'nan']

# 5G概念股
stocks1 = list(set(richinx+data_5g))
# 興櫃
stocks2 = ['5222.TWO']
stocks = list(set(stocks1+stocks2))
codes = stocks + bonds

# 爬蟲 開始日期start
#start = datetime.datetime(2012,1,1)
#end = datetime.datetime(2018,12,31)
#data_price = web.DataReader(codes, 'yahoo', start=start, end = end)

# 債券代碼
tar_bond = '00679B.TWO'

# 挑出調整後收盤價
#df = data_price['Adj Close']
#df.to_excel(join(out,'adj close2.xlsx'))
df = pd.read_excel(join(out,'adj close2.xlsx'), index_col = 0)

df1 = df.dropna(how='all',axis = 1)
df1 = df1[[not math.isnan(x) for x in df1[tar_bond]]]
df1 = df1.dropna(how='any',axis = 1)
df1 = df1.dropna()
#df.plot.line()

# 一年天數
N = 252

# 計算年化return
data_return = np.log(df1.shift(N)/df1)

# 挑選前半年sharpe ratio最高的前五檔標的
data_return = data_return.dropna()

return_EDA = pd.DataFrame(data_return.mean(), columns = ['return'])
return_EDA['std'] = data_return.std()
return_EDA['bench'] = return_EDA['return']/return_EDA['std']
return_EDA = return_EDA.sort_values('bench', ascending = False)
targets = list(return_EDA.index[:5])


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

# 建立隨機投資組合
def random_portfolios(num_portfolios, mean_returns, cov_matrix):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(len(targets))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = Portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return) / portfolio_std_dev
    return results, weights_record

def profolios_allocation(mean_returns, cov_matrix, num_portfolios):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='PuBuGn', marker='o', s=10, alpha=0.3)
#    plt.colorbar()
    plt.title('Profolios Performance Allocation', fontsize=20,fontweight='bold',)
    plt.xlabel('Standard Deviation')
    plt.ylabel('Average Return')
    plt.legend(labelspacing=0.8)
    plt.show()
    
#找到相同投資組合報酬率下最小的風險
def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return Portfolio_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(Portfolio_volatility, num_assets*[1/num_assets,], args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)
    return result

#找到效率前緣樣本
def efficient_profolios(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_efficient_frontier(mean_returns, cov_matrix, num_portfolios):
    results, _ = random_portfolios(num_portfolios,mean_returns, cov_matrix)
     
    #點出所有投組報酬率及風險
    mpl_fig = plt.figure(figsize=(9, 5))
#    plt.figure(figsize=(9, 5))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGn', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    
    #找出最小風險投資組合及其報酬率
    min_vol = min_variance(mean_returns, cov_matrix)
    sdp_min, rp_min = Portfolio_performance(min_vol['x'], mean_returns, cov_matrix)
    
    #點出最小風險之投資組合
    plt.scatter(sdp_min,rp_min,marker='*',color='red',s=500, label='Min Volitility')

    #畫出所有投組樣本的效率前緣線    
    target = np.linspace(rp_min, mean_returns.max(), 50)
    efficient_portfolios = efficient_profolios(mean_returns, cov_matrix, target)
    plt.plot([p['fun'] for p in efficient_portfolios], 
             target, linestyle='-', color='coral',linewidth=4, label='Efficient Frontier') 
    plt.title('Efficient Frontier on Profolios Allocation', fontsize = 18)
    plt.xlabel('Standard Deviation', fontsize = 18)
    plt.ylabel('Expected Return', fontsize = 18)
    plt.legend(labelspacing=0.9 ,loc ='best')
    plt.savefig(join(out,'Efficient Frontier.png'))
    
    plotly_fig = tls.mpl_to_plotly(mpl_fig)
    plotly_fig.layout.width = 1080
    plotly_fig.layout.height = 600
    plotly.offline.plot(plotly_fig, filename = join(out,'Efficient Frontier.html'))

#[x in data_5g for x in targets] #stocks1、stocks2、data_5g、bonds
fudata = web.DataReader(targets, 'yahoo', start=datetime.datetime(2019,1,1), end = datetime.datetime(2019,9,29))
fudata = fudata['Adj Close'].dropna()
#fudata.index = fudata.index.strftime("%Y/%m/%d")
#
#fudata.to_excel(join(out, 'targets data (2019.1.2~2019.9.30).xlsx'), index_label = 'date')
#fudata = pd.read_excel(join(out, 'targets data (2019.1.2~2019.9.30).xlsx'),index_col = 0)
mdata = pd.read_excel(join(src, 'TWO.xlsx'), index_col = 0)
mdata = mdata.loc[fudata.index,:]

# 標的前半年log return
tarreturn = data_return[-int(N/2):][targets]
#tarreturn.plot.line()

# plot return
line_data = []
for col in tarreturn:
    plot = go.Scatter(
        x = tarreturn.index,
        y = tarreturn[col],
        mode = 'lines',
        name = col
        )
    line_data.append(plot)
    
layout = dict(title = 'Returns of Targets'
             )

fig = dict(data = line_data , layout = layout)
plotly.offline.plot(fig , filename = join(out, 'Returns of Targets.html'), image='png')


# 標的前半年變異數
Var = tarreturn.var()

# 平均報酬
mean_returns = tarreturn.mean()

# 共變異數矩陣
cov_matrix = tarreturn.cov()

# 最佳化權重
w = min_variance(mean_returns , cov_matrix)["x"]

# expected return
exp_r = sum(mean_returns*w)

# 自訂投資組合數
num_portfolios = 30000

# 畫效率前緣
display_efficient_frontier(mean_returns, cov_matrix, num_portfolios)

'''樣本外績效'''
# 投入金額
fund = 20000000
c_ratio = .2
cash = fund * c_ratio
V0 = fund * (1-c_ratio)

# 每檔股票投入金額
P0 = w * V0

# 每檔買入單位數
U0 = P0/fudata.iloc[0,:]
UM = V0/mdata.iloc[0,:]

# 每日投組價值
V = []
V_cash = []
VM = []
for i in fudata.index:
    V.append(sum(U0*fudata.loc[i,:]))
    V_cash.append(sum(U0*fudata.loc[i,:]) + cash)
    VM.append(sum(UM*mdata.loc[i,:]))

# 年化報酬率
R = pow(V[-1]/V0,12/9) - 1
RM = pow(VM[-1]/V0,12/9) - 1

Std = np.std(V)

'''plot'''
plot_v = go.Scatter(
        x = fudata.index,
        y = V,
        mode = 'lines',
        name = 'Value of Portfolio'
        )

plot_v_c = go.Scatter(
        x = fudata.index,
        y = V_cash,
        mode = 'lines',
        name = 'Value of Portfolio + cash'
        )

plot_m = go.Scatter(
        x = fudata.index,
        y = VM,
        mode = 'lines',
        name = 'TWO Market'
        )

layout1 = go.Layout(title = 'Trace of Portfolio Value',
               xaxis = dict(title='Date'),
               yaxis = dict(title='Value')
             )

fig1 = dict(data = [plot_v,plot_v_c,plot_m] , layout = layout1)
plotly.offline.plot(fig1 , filename = join(out, 'Value of Portfolio.html'), image='png')

