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
            ("LSPI", "LSPI_Grouped_DTW"): "experiment_101_LSPI_Normal_Grouped_KMedoids_DTW_",
            ("LSPI", "LSPI_Pooled"): "experiment_102_LSPI_Normal_Pooled_KMedoids_DTW_",
            ("LSPI", "LSPI_Separate"): "experiment_103_LSPI_Normal_Separate_KMedoids_DTW",
            ("LSPI", "LSPI_Grouped_Benchmark"): "experiment_104_LSPI_Normal_Grouped_benchmark_",
            ("Q-Learning", "Q_Grouped_DTW"): "experiment_105_Q_Normal_Grouped_KMedoids_DTW_",
            ("Q-Learning", "Q_Pooled"): "experiment_106_Q_Normal_Pooled_KMedoids_DTW_",
            ("Q-Learning", "Q_Separate"): "experiment_107_Q_Normal_Separate_KMedoids_DTW_",
            ("Q-Learning","Q_Grouped_Benchmark"): "experiment_110_Q_Normal_Grouped_benchmark_"
            }

#db_names = {
#            ("LSPI", "LSPI_Grouped_DTW"): "experiment_103_LSPI_Normal_Grouped_KMedoids_DTW",
#            ("Q","LSPI_Pooled"): "experiment_118_LSPI_Normal_Grouped_KM_DTW_perfect"
#}

col_names = ['Algorithm', 'Setup', 'Average_daily_reward']
my_df = pandas.DataFrame(columns=col_names)

for key, val in db_names.items():

    db_name = val
    db = db_connection[db_name]
    query_result = db.agent_state.find(query, projection)

    print(query_result[:])
    result = np.zeros(shape=(query_result.count(), int((len(query_result[0]["rewards"])) / 24)))
    print(result.shape)

    ndays = result.shape[1]

    for i in range(0, query_result.count()):
        y = 0
        result = []
        for e in range(0, ndays):
            # Reward
            res = (query_result[i]["rewards"][y:y + 24])
            res = [float(i) for i in res]
            #result = result.append(np.mean(res))
            y = y + 24
            my_df.loc[len(my_df)] = [key[0], key[1], np.mean(res)]


print(my_df)
ax = sns.barplot(y="Average_daily_reward", x="Algorithm", hue='Setup', data=my_df)
plt.xlabel("Algorithm")
plt.ylabel("Average daily reward")
plt.title("Average daily rewards over different learning setups") # You can comment this line out if you don't need title
fig = ax.get_figure()
plt.show(fig)
fig.savefig('bar_plot.png')