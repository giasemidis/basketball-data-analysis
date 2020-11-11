import glob
import numpy as np
import pandas as pd

import plotly_express as px

from utils import make_scatter_plot, make_bar_plot, make_scatter_plot_at_least_n_points


# %% Load Data

files_pattern = 'data/euroleague_results*csv'
data_list_files = glob.glob(files_pattern)

df = pd.concat([pd.read_csv(f) for f in data_list_files], ignore_index=True)
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
df_flat['Season_int'] = df_flat['Season'].apply(lambda x: int(x[-4:]))

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

# %% Plots: Home/Away Scores

fig = px.box(df_flat, x="Season", y="Score", color="Loc", notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
fig.show()

fig = px.box(df, x="Season", y="Home Score", color="Game Result", notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
fig.show()

fig = px.box(df, x="Season", y="Away Score", color="Game Result", notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
fig.show()

fig = px.box(df, x="Season", y="Score Difference", color="Game Result",
             notched=True)
fig.layout.yaxis.update({'showgrid': True, 'gridcolor': 'rgb(200, 200, 200)'})
fig.show()


# %% Scatter plots - probability of winning when scoring at least N points
make_scatter_plot_at_least_n_points([df_flat, df_flat[df_flat['Loc'] == 'Home'],
                                     df_flat[df_flat['Loc'] == 'Away']],
                                    ['All', 'Home', 'Away'])

make_scatter_plot_at_least_n_points([df_flat[df_flat['Season_int'] == 2017],
                                     df_flat[df_flat['Season_int'] == 2018],
                                     df_flat[df_flat['Season_int'] == 2019]],
                                    ['2017', '2018', '2019'])

# %% Scatter plots - probability of winning when scoring points in a range.

# %% Bar plots

make_bar_plot([df_flat, df_flat[df_flat['Loc'] == 'Home'],
               df_flat[df_flat['Loc'] == 'Away']], ['All', 'Home', 'Away'])

make_bar_plot([df_flat[df_flat['Season_int'] == 2017],
               df_flat[df_flat['Season_int'] == 2018],
               df_flat[df_flat['Season_int'] == 2019]],
              ['2017', '2018', '2019'])

# %% Scatter plots

make_scatter_plot([df_flat, df_flat[df_flat['Loc'] == 'Home'],
                   df_flat[df_flat['Loc'] == 'Away']],
                  ['All', 'Home', 'Away'])

make_scatter_plot([df_flat[df_flat['Season_int'] == 2017],
                   df_flat[df_flat['Season_int'] == 2018],
                   df_flat[df_flat['Season_int'] == 2019]],
                  ['2017', '2018', '2019'])
