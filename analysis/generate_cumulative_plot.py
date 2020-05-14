#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from pymongo import MongoClient
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

matplotlib.use('agg')


db_connection = MongoClient('localhost', 27017)
query = {'rewards': {'$exists': 1}}
projection = {'_id': 0, 'rewards': 1}
sort = [('agent_id',1)]

db_names = {("LSPI", "LSPI_Grouped_DTW"): "experiment_101_LSPI_Normal_Grouped_KMedoids_DTW_2",
            ("LSPI", "LSPI_Pooled"): "experiment_102_LSPI_Normal_Pooled_KMedoids_DTW_2",
            ("LSPI", "LSPI_Separate"): "experiment_103_LSPI_Normal_Separate_KMedoids_DTW_2",
            ("LSPI", "LSPI_Grouped_Benchmark"): "experiment_104_LSPI_Normal_Grouped_benchmark_2",
            ("Q-Learning", "Q_Grouped_DTW"): "experiment_105_Q_Normal_Grouped_KMedoids_DTW_2",
            ("Q-Learning", "Q_Pooled"): "experiment_106_Q_Normal_Pooled_KMedoids_DTW_2",
            ("Q-Learning", "Q_Separate"): "experiment_107_Q_Normal_Separate_KMedoids_DTW_2",
            ("Q-Learning","Q_Grouped_Benchmark"): "experiment_108_Q_Normal_Grouped_benchmark_2"}

db_names = {("LSPI", "LSPI_Grouped_DTW"): "experiment_103_LSPI_Normal_Grouped_KMedoids_DTW",
            ("Q", "Q_grouped"): "experiment_101_LSPI_Normal_Grouped_KMedoids_DTW"
            }

col_names = ['Day',  'Setup', 'Cumulative_daily_reward']
my_df = pandas.DataFrame(columns=col_names)

for key, val in db_names.items():
    db_name = val
    db = db_connection[db_name]
    query_result = db.agent_state.find(query, projection)
    ndays = int((len(query_result[0]["rewards"]))/24)
    print(query_result[:])
    running_count_reward = 0

    for e in range(0, ndays):
        y = 0
        result = []
        for i in range(0,query_result.count()):
            res = (query_result[i]["rewards"][y:y + 24])
            res = [float(i) for i in res]
            y = y + 24
            running_count_reward += np.mean(res)
        my_df.loc[len(my_df)] = [e, key[1], running_count_reward]


# Create plot
sns.set_style("ticks")

ax = sns.tsplot(data=my_df,
                time="Day",
                unit="Setup",
                condition="Setup",
                value="Cumulative_daily_reward")

sns.tsplot(data=my_df,
           time='Day', unit='Setup',
           condition='Setup',
           value='Cumulative_daily_reward')

plt.xlabel("Algorithm")
plt.ylabel("Average daily reward")
plt.title("Average daily rewards over different learning setups")
fig = ax.get_figure()
plt.show(fig)
fig.savefig('cumulative_plot.png')

fig, ax = plt.subplots()
for key, grp in my_df.groupby(['Setup']):
    ax = grp.plot(ax=ax, kind='line', x='Day', y='Cumulative_daily_reward', label=key)

plt.legend(loc='best')
plt.show()
fig2 = ax.get_figure()
fig2.savefig('cumulative_plot_2.png')