import numpy as np
from pymongo import MongoClient
from pymongo.errors import *
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

client = MongoClient('mongodb://localhost:27017/')


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

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 1 -accumilated_reward_Grouped
number_of_agents = 100
number_of_days = 107
experiment = 'experiment_101_LSPI_Normal_Grouped_KMedoids_DTW_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Grouped_LSPI = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")

################################################### 2- experiment_102_LSPI_Normal_Pooled_KMedoids_DTW

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_102_LSPI_Normal_Pooled_KMedoids_DTW_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Pooled_LSPI = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")
################################################### 3- experiment_103_LSPI_Normal_Separate_KMedoids_DTW

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_103_LSPI_Normal_Separate_KMedoids_DTW_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Separate_LSPI = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")
################################################### 4- experiment_109_LSPI_Normal_Grouped_benchmark

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_104_LSPI_Normal_Grouped_benchmark_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Bench_LSPI = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")


################################################### 5- experiment_105_Q_Normal_Grouped_KMedoids_DTW_2

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_105_Q_Normal_Grouped_KMedoids_DTW_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Grouped_Q = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")

################################################### 5- experiment_106_Q_Normal_Pooled_KMedoids_DTW_2

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_106_Q_Normal_Pooled_KMedoids_DTW_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Pooled_Q = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")


################################################### 5- experiment_107_Q_Normal_Separate_KMedoids_DTW_2

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_107_Q_Normal_Separate_KMedoids_DTW_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Separate_Q = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")

################################################### 5- experiment_108_Q_Normal_Grouped_benchmark_2

number_of_agents = 100
number_of_days = 107
experiment = 'experiment_108_Q_Normal_Grouped_benchmark_2'

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

db = client[experiment]

collection = db['agent_state']
query_result = db.agent_state.find({}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(map(float, query_result[agent_id - 1]["rewards"]))[
                                                0:number_of_days * 24]
    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])

accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]
for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

accumilated_reward_Bench_Q = accumilated_reward_ShrinkingCentral
print("Separate 24 Finished ")
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

end_day = 107

plt.figure(figsize=[12, 10])

fig, plt1 = plt.subplots()

plt1.plot(accumilated_reward_Grouped_LSPI[0:end_day])
plt1.plot(accumilated_reward_Pooled_LSPI[0:end_day])
plt1.plot(accumilated_reward_Separate_LSPI[0:end_day])
plt1.plot(accumilated_reward_Bench_LSPI[0:end_day])

plt1.legend([
    "Gouped",
    "Pooled",
    "Separate",
    "Benchmark grouped"
], loc = 1)

plt.ylabel('Cummulative Reward')
plt.xlabel('Day')
#plt.xlim([0, 107])
axins = zoomed_inset_axes(plt1, 2.5, loc=2) # zoom-factor: 2.5, location: upper-left
axins.plot(accumilated_reward_Grouped_LSPI[0:end_day])
axins.plot(accumilated_reward_Pooled_LSPI[0:end_day])
axins.plot(accumilated_reward_Separate_LSPI[0:end_day])
axins.plot(accumilated_reward_Bench_LSPI[0:end_day])

x1, x2, y1, y2 = 0, 15, 0, 200
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.xticks(visible=False)
plt.yticks(visible=False)

plt_2 = mark_inset(plt1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
fig.savefig('cumulative_plot_LSPI.png')
print("FINISHED")




fig.clf()

plt.figure(figsize=[12, 10])

fig, plt1 = plt.subplots()

plt1.plot(accumilated_reward_Grouped_Q[0:end_day])
plt1.plot(accumilated_reward_Pooled_Q[0:end_day])
plt1.plot(accumilated_reward_Separate_Q[0:end_day])
plt1.plot(accumilated_reward_Bench_Q[0:end_day])
plt1.legend([
    "Gouped",
    "Pooled",
    "Separate",
    "Benchmark grouped"

], loc = 1)
plt.ylabel('Cummulative Reward')
plt.xlabel('Day')

axins = zoomed_inset_axes(plt1, 2.5, loc=2) # zoom-factor: 2.5, location: upper-left
axins.plot(accumilated_reward_Grouped_Q[0:end_day])
axins.plot(accumilated_reward_Pooled_Q[0:end_day])
axins.plot(accumilated_reward_Separate_Q[0:end_day])
axins.plot(accumilated_reward_Bench_Q[0:end_day])

x1, x2, y1, y2 = 0, 15, 0, 75
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

plt.xticks(visible=False)
plt.yticks(visible=False)
plt_2 = mark_inset(plt1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
fig.savefig('cumulative_plot_Q.png')
print("FINISHED")
