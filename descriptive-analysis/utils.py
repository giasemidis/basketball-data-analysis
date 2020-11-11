import numpy as np
import plotly.graph_objs as go


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


def find_probs_at_least_n_points(df, step=1):
    '''
    Probability of wining when scoring at least N points
    '''
    min_value = step * (df['Score'].min() // step)
    max_value = step * (df['Score'].max() // step)
    x = np.arange(min_value, max_value + 1, step, dtype=int)
    prob = np.zeros(x.shape[0])
    for i, u in enumerate(x[:-1]):
        num = ((df['Score'] >= u) & (df['Team Result'] == 'W')).sum()
        den = (df['Score'] >= u).sum()
        prob[i] = num / den
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
    fig.show()
    return


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
    fig.show()
    return


def make_scatter_plot_at_least_n_points(dfs, names, title=''):
    data = []
    for df, name in zip(dfs, names):
        prob, x = find_probs_at_least_n_points(df, 5)
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
    fig.show()
    return
