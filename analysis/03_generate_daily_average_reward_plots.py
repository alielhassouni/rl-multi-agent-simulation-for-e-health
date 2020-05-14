from pymongo import MongoClient
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas

db_connection = MongoClient('localhost', 27017)
query = {'rewards': {'$exists': 1}}
projection = {'_id': 0, 'rewards': 1}
sort = [('agent_id',1)]

db_names = {
            ("LSPI", "LSPI_Grouped_DTW"): "experiment_101_LSPI_Normal_Grouped_KMedoids_DTW_2",
            ("LSPI", "LSPI_Pooled"): "experiment_102_LSPI_Normal_Pooled_KMedoids_DTW_2",
            ("LSPI", "LSPI_Separate"): "experiment_103_LSPI_Normal_Separate_KMedoids_DTW_2",
            ("LSPI", "LSPI_Grouped_Benchmark"): "experiment_104_LSPI_Normal_Grouped_benchmark_2",
            ("Q-Learning", "Q_Grouped_DTW"): "experiment_105_Q_Normal_Grouped_KMedoids_DTW_2",
            ("Q-Learning", "Q_Pooled"): "experiment_106_Q_Normal_Pooled_KMedoids_DTW_2",
            ("Q-Learning", "Q_Separate"): "experiment_107_Q_Normal_Separate_KMedoids_DTW_2",
            ("Q-Learning","Q_Grouped_Benchmark"): "experiment_108_Q_Normal_Grouped_benchmark_2"
}

db_names = {
            ("LSPI", "LSPI_Grouped_DTW"): "experiment_103_LSPI_Normal_Grouped_KMedoids_DTW"}

col_names = ['Day', 'Setup', 'Average_daily_reward', 'sd']
my_df = pandas.DataFrame(columns=col_names)

for key, val in db_names.items():

    db_name = val
    db = db_connection[db_name]
    query_result = db.agent_state.find(query, projection)
    result = np.zeros(shape=(query_result.count(), int((len(query_result[0]["rewards"]))/24)))
    ndays = result.shape[1]
    y = 0

    for e in range(0, ndays):
                running_mean = 0
                means = []
                for i in range(0, query_result.count()):
                    res = (query_result[i]["rewards"][y:y+24])
                    res = [float(i) for i in res]
                    running_mean += np.mean(res)
                    means.append(np.mean(res))
                y = y + 24

                my_df.loc[len(my_df)] = [e, key[1], running_mean, np.std(means)]

sns.set(color_codes=True)
my_df2 = my_df.pivot(index='Day', columns='Setup', values='Average_daily_reward')

from ggplot import *
print(my_df)
#g = ggplot(aes(x='Day', y='Average_daily_reward', colour='Setup'), data=my_df) + geom_line()
#g.save('output2_2gg.png')



col_names = ['Day', 'Setup', 'Average_daily_reward']
my_df_ts = pandas.DataFrame(columns=col_names)

for key, val in db_names.items():

    db_name = val
    db = db_connection[db_name]
    query_result = db.agent_state.find(query, projection)
    result = np.zeros(shape=(query_result.count(), int((len(query_result[0]["rewards"]))/24)))
    ndays = result.shape[1]

    for i in range(0, query_result.count()):

                running_mean = 0
                means = []
                y = 0
                for e in range(0, ndays):

                    res = (query_result[i]["rewards"][y:y+24])
                    res = [float(i) for i in res]
                    running_mean += np.mean(res)
                    #means.append(np.mean(res))
                    y = y + 24

                    my_df_ts.loc[len(my_df_ts)] = [e, key[1], np.mean(res)]



print(my_df_ts)
#ax = my_df2.plot(y="LSPI_Grouped_DTW", err_style = "ci", legend=False, color="r")
#ax2 = ax.twinx()
#my_df2.plot(y="LSPI_Pooled", ax=ax, err_style="ci", legend=False, color="g")
#my_df2.plot(y="LSPI_Separate", ax=ax,  err_style="ci", legend=False, color="b")
#my_df2.plot(y="LSPI_Grouped_Benchmark", ax=ax,  err_style="ci", legend=False, color="k")

#ax.figure.legend()
#plt.show()


#fig = ax.get_figure()
#fig.savefig('output2_2_ts.png')

#g = sns.tsplot(my_df, hue='Setup', size=5, aspect=1.5)
#g.map(plt.plot, 'Day', 'Average_daily_reward').add_legend()
#g.ax.set(xlabel='Day',
#         ylabel='Average_daily_reward',
#         title='Average_daily_reward')
#g.fig.autofmt_xdate()
#fig = g.get_figure()
#fig.savefig('output2_2.png')

#data = (query_result[1]["rewards"])
#data = [float(i) for i in data]
#print(data)

#my_df_ts = my_df_ts.reset_index().pivot_table(values='Average_daily_reward', index='Day', columns='Setup', aggfunc='mean')

#rowdicts = []

#for l, d in my_df_ts.groupby("Day Setup".split()):
#    d = {"Day": l[0], "Setup": l[1]}
#    rowdicts.append(d)

#my_df_ts = pandas.DataFrame.from_dict(rowdicts)
my_df_ts

#my_df_ts['DayTime'] = my_df_ts.index
#my_df_ts = pandas.DataFrame.from_dict(my_df_ts)
print(my_df_ts)
#sns_plot = sns.tsplot(data=my_df_ts, time = 'Day', values = 'LSPI_Grouped_DTW', hue = 'Setup', ci=[ 95])
#sns_plot = sns.tsplot(data=my_df_ts, ci="sd")

sns.set(color_codes=True)
sns_plot = sns.tsplot(data=my_df_ts, time = 'Day', value='Average_daily_reward', err_style = 'ci_band', unit = 'Average_daily_reward')

#sns_plot = sns.tsplot(data=result)
#sns_plot = sns.tsplot(data=result, estimator=np.median)
#sns_plot = sns.tsplot(data=result, ci=[ 95])
#sns_plot = sns.tsplot(data=my_df, err_style="boot_traces", n_boot=500)

fig = sns_plot.get_figure()
fig.savefig('output_ts_plot.png')