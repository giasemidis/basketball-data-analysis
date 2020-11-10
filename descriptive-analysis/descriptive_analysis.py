# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:58:39 2019

@author: giase
"""

import numpy as np
import pandas as pd
import sys

import plotly_express as px
from plotly.offline import plot
import plotly.graph_objs as go

sys.path.append('auxiliary/')


def find_probs(df):
    '''
    Probability of Winning when scoring in interval [x, x+4]
    '''
    min_value = (df['Score'].min() // 5) * 5
    max_value = (df['Score'].max() // 5 + 1) * 5
    x = np.arange(min_value, max_value + 1, 5, dtype=int)
    prob = np.zeros(x.shape[0] - 1)
    for i, u in enumerate(x[:-1]):
        ii = (df['Score'] >= x[i]) & (df['Score'] < x[i + 1])
        num = np.sum((df[ii]['Team Result'] == 'W'))
        den = np.sum(ii)
        prob[i] = num / den if den > 0 else 0
    return prob, x[:-1]


def make_x_interv(x):
    ans = [str(x[i]) + '-' + str(x[i + 1] - 1) for i in range(x.shape[0] - 1)]
    ans.append(str(x[-1]) + '-' + str(x[-1] + 4))
    return ans


def make_bar_plot(dfs, names, title=''):
    data = []
    for df, name in zip(dfs, names):
        prob, x = find_probs(df)
        data.append(go.Bar(x=make_x_interv(x), y=prob, name=name))

    layout = go.Layout(title=title,
                       xaxis={'title': 'Score', 'showgrid': True},
                       yaxis={'title': 'Probability', 'showgrid': True})
    fig = go.Figure(data, layout)
    plot(fig)


def make_scatter_plot(dfs, names, title=''):
    data = []
    for df, name in zip(dfs, names):
        prob, x = find_probs(df)
        data.append(go.Scatter(x=x, y=prob,
                               # fill='tozeroy',
                               line={'shape': 'hv'},
                               name=name))

    layout = go.Layout(title=title,
                       xaxis={'title': 'Score',
                              'showgrid': True,
                              'gridcolor': 'rgb(200, 200, 200)',
                              'type': 'category'},
                       yaxis={'title': 'Probability',
                              'showgrid': True,
                              'gridcolor': 'rgb(200, 200, 200)'})
    fig = go.Figure(data, layout)
    plot(fig)


# %% load data

df1 = pd.read_csv('data/euroleague_results_2016_2017.csv')
df1.insert(1, 'Season', 2017)
df2 = pd.read_csv('data/euroleague_results_2017_2018.csv')
df2.insert(1, 'Season', 2018)
df3 = pd.read_csv('data/euroleague_results_2018_2019.csv')
df3.insert(1, 'Season', 2019)

df = pd.concat([df1, df2, df3], ignore_index=False)
df.reset_index(drop=True, inplace=True)

df['Game Result'] = np.where(df['Home Score'] > df['Away Score'], 1, 2)
df['Score Difference'] = np.abs(df['Home Score'] - df['Away Score'])

# %% Reshape the data

df_flat = pd.melt(df, id_vars=['Season', 'Round', 'Game Result'],
                  value_vars=['Home Score', 'Away Score'],
                  var_name='Loc', value_name='Score')
df_flat['Loc'] = df_flat['Loc'].apply(lambda x: x.split(' ')[0])
df_flat['Team Result'] = np.where(((df_flat['Game Result'] == 1) &
                                   (df_flat['Loc'] == 'Home')) |
                                  ((df_flat['Game Result'] == 2) &
                                   (df_flat['Loc'] == 'Away')), 'W', 'L')

# %% Stat Table

dfgroup = df_flat.groupby(['Season', 'Loc'])['Score'].mean().unstack('Loc')
dfgroup.columns = ['Away Mean Score', 'Home Mean Score']
dfgroup.reset_index(inplace=True)

dff = (df.groupby(['Season', 'Game Result'])['Game Result'].
       count().unstack('Game Result'))
dff.columns = ['Home Wins', 'Away Wins']
dff.reset_index(inplace=True)

dfgroup = dfgroup.merge(dff, on='Season')

dff = (df_flat.groupby(['Season', 'Game Result'])['Score'].
       mean().unstack('Game Result'))
dff.columns = ['Home Win Mean Score', 'Away Win Mean Score']
dff.reset_index(inplace=True)

dfgroup = dfgroup.merge(dff, on='Season')

print(dfgroup)

# %% Stat Plots

fig = px.box(df_flat, x="Season", y="Score", color="Loc", notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
plot(fig, 'plot0')

fig = px.box(df, x="Season", y="Home Score", color="Game Result", notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
plot(fig, 'plot1')

fig = px.box(df, x="Season", y="Away Score", color="Game Result", notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
plot(fig, 'plot2')

fig = px.box(df, x="Season", y="Score Difference", color="Game Result",
             notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
plot(fig, 'plot3')

# %% Bar plots

make_bar_plot([df_flat, df_flat[df_flat['Loc'] == 'Home'],
               df_flat[df_flat['Loc'] == 'Away']], ['All', 'Home', 'Away'])

make_bar_plot([df_flat[df_flat['Season'] == 2017],
               df_flat[df_flat['Season'] == 2018],
               df_flat[df_flat['Season'] == 2019]],
              ['2017', '2018', '2019'])

# %% Scatter plots

make_scatter_plot([df_flat, df_flat[df_flat['Loc'] == 'Home'],
                   df_flat[df_flat['Loc'] == 'Away']],
                  ['All', 'Home', 'Away'])

make_scatter_plot([df_flat[df_flat['Season'] == 2017],
                   df_flat[df_flat['Season'] == 2018],
                   df_flat[df_flat['Season'] == 2019]],
                  ['2017', '2018', '2019'])
