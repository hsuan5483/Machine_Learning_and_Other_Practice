#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Pei-Hsuan Hsu
"""

import numpy as np
import pickle as pk
import plotly
import plotly.graph_objs as go

file_path = 'path of .py'
dataset = pk.load(open(file_path + 'TWN.dat' , 'rb'))
savepath = 'Plots/'

### PSY
'''
==========
心理線 PSY
==========
PSY = 有上漲的天數( N日內 ) / N * 100
20 ~ 80之間移動時為盤整狀態
數值低於10時則可能出現反彈機會
數值若高於90以上時，則可能短線過熱

應用
1. 一般心理線介於25%~75%是合理變動範圍。

2. 超過75%或低於25%，就有買超或賣超現象，股價回跌或回升機會增加，
   此時可準備賣出或買進。在大多頭或大空頭市場初期，
   可將超買、超賣點調整至83%、17%值到行情尾聲，再調回70%、25%。
   
3. 當行情出現低於10%或高於90%時；是真正的超賣和超買現象，
   行情反轉的機會相對提高，此時為賣出和買進時機。
'''

#上漲天數
dataset['rise'] = np.sign(dataset['TWII'] - dataset['TWII'].shift(1)).replace(-1,0)

#6 days PSY
dataset['PSY_6'] = dataset['rise'].rolling(6).mean()*100
psy_signal = dataset['PSY_6'] < 10
psy_signal &= (psy_signal.shift().rolling(20).mean() == 0)
dataset['psy_long'] = psy_signal

dataset = dataset.dropna()

hold = 5
profit_psy = []
DATE = []
for date in range(len(dataset.index)) :
    if dataset.ix[date , 'psy_long'] == True :
        DATE.append(dataset.index[date])
        buy = dataset.ix[date , 'TWII']
        sell = dataset.ix[date + hold , 'TWII']
        profit_psy.append((sell - buy)/buy)

psy_profit = sum(profit_psy)

'''
===========
TWII vs PSY
===========
'''

### 畫圖
trace_twii = go.Scatter(
    x = dataset.index.strftime("%Y-%m-%d"),
    y = dataset['TWII'],
    name = "TwII",
    line = dict(color = '#616263'),
    )

trace_psy = go.Scatter(
    x = dataset.index.strftime("%Y-%m-%d"),
    y = dataset['PSY_6'] ,
    name = "PSY(6 days)",
    line = dict(color = '#98C7FB'),
    yaxis = 'y2'
    )

data = [trace_twii , trace_psy]

layout = go.Layout(
    title ='TWII vs PSY(6 days)',
    yaxis = dict(
        title ='TWII'
    ),
    yaxis2 = dict(
        title ='PSY(6 days)',
        titlefont = dict(
            color = '#98C7FB'
        ),
        tickfont = dict(
            color = '#98C7FB'
        ),
        overlaying = 'y',
        side = 'right'
    )
)


fig = dict(data = data, layout = layout)

#將圖表存至帳號
plotly.offline.plot(fig , filename = savepath + "TWII vs PSY.html")


'''
==========
long point
==========
'''
### 畫圖

trace_psy_long = go.Scatter(
    x = dataset.index.strftime("%Y-%m-%d"),
    y = psy_signal.astype(float),
    name = "long point",
    line = dict(color = '#53A0F7'),
    yaxis = 'y2'
    )

data = [trace_twii , trace_psy_long]

layout = go.Layout(
    title ='TWII vs Long Point (PSY)',
    yaxis = dict(
        title ='TWII'
    ),
    yaxis2 = dict(
        tickfont = dict(
            color = '#53A0F7'
        ),
        overlaying = 'y',
        side = 'right'
    )
)


fig = dict(data = data, layout = layout)

#將圖表存至帳號
plotly.offline.plot(fig , filename = savepath + "TWII vs Long Point (PSY).html")


'''
=====
TWNVIX 交易策略
跌破30 買進
=====
'''

vix_signal = dataset['TWNVIX'] < 28
vix_signal &= (vix_signal.shift().rolling(20).mean() == 0)
dataset['vix_signal'] = vix_signal
dataset['vix_level'] = 28

dataset = dataset.dropna()

hold = 15
profit_vix = []
DATE1 = []
for date in range(len(dataset.index)) :
    if dataset.ix[date , 'vix_signal'] == True :
        DATE1.append(dataset.index[date])
        buy = dataset.ix[date , 'TWII']
        sell = dataset.ix[date + hold , 'TWII']
        profit_vix.append((sell - buy)/buy)

vix_profit = sum(profit_vix)


#plots
trace_twnvix = go.Scatter(
    x = dataset.index.strftime("%Y-%m-%d"),
    y = dataset['TWNVIX'],
    name = "TWNVIX",
    line = dict(color = '#F4BC70'),
    opacity = 0.8,
    yaxis = 'y2')

trace_vix_level = go.Scatter(
    x = dataset.index.strftime("%Y-%m-%d"),
    y = dataset['vix_level'],
    name = "TWNVIX = 28",
    line = dict(color = '#F48470'),
    opacity = 0.8,
    yaxis = 'y2')

data = [trace_twii , trace_twnvix , trace_vix_level]
layout = go.Layout(
    title ='TWII vs TWNVIX',
    yaxis = dict(
        title ='TWII'
    ),
    yaxis2 = dict(
        title ='TWNVIX',
        titlefont = dict(
            color = '#F4BC70'
        ),
        tickfont = dict(
            color = '#F4BC70'
        ),
        overlaying = 'y',
        side = 'right'
    )
)


fig = dict(data = data, layout = layout)

plotly.offline.plot(fig , filename = savepath + "TWII vs TWNVIX.html")


'''
===============
long point(vix)
===============
'''
### 畫圖

trace_vix_long = go.Scatter(
    x = dataset.index.strftime("%Y-%m-%d"),
    y = vix_signal.astype(float),
    name = "long point",
    line = dict(color = '#FE9328'),
    yaxis = 'y2'
    )


data = [trace_twii , trace_vix_long]

layout = go.Layout(
    title ='TWII vs Long Point (VIX)',
    yaxis = dict(
        title ='TWII'
    ),
    yaxis2 = dict(
        tickfont = dict(
            color = '#FE9328'
        ),
        overlaying = 'y',
        side = 'right'
    )
)


fig = dict(data = data, layout = layout)

plotly.offline.plot(fig , filename = savepath + "TWII vs Long Point (VIX).html")

