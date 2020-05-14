#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from pymongo import MongoClient
import matplotlib
import numpy as np
import seaborn as sns
from itertools import chain

matplotlib.use('agg')
sns.set(color_codes=True)

db_connection = MongoClient('localhost', 27017)
db_name = "experiment_103_LSPI_Normal_Grouped_KMedoids_DTW"
db = db_connection[db_name]

# Query
query = {'rewards': {'$exists': 1}}
projection = {'_id': 0, 'rewards': 1}
sort = [('agent_id',1)]
query_result = db.agent_state.find(query, projection)

number_of_agents = 50
number_of_days = 67

rewards_allAgents_hourly = np.zeros([number_of_agents, number_of_days * 24])
rewards_allAgents_daily = np.zeros([number_of_agents, number_of_days])
rewards_all_daily_ShrinkingCentral = np.zeros([number_of_days, 1])
accumilated_reward_ShrinkingCentral = np.zeros([number_of_days, 1])

for agent_id in range(1, number_of_agents + 1):
    rewards_allAgents_hourly[agent_id - 1, :] = list(
        map(
            float, query_result[agent_id - 1]["rewards"]))[0:number_of_days * 24]

    for day in range(0, number_of_days):
        rewards_allAgents_daily[agent_id - 1, day] = np.sum(
            rewards_allAgents_hourly[agent_id - 1, 24 * day: 24 * day + 24])

rewards_all_daily_ShrinkingCentral = np.mean(rewards_allAgents_daily, axis=0)
rewards_all_daily_ShrinkingCentral = rewards_all_daily_ShrinkingCentral.reshape([number_of_days, 1])
accumilated_reward_ShrinkingCentral[0, 0] = rewards_all_daily_ShrinkingCentral[0, 0]

for i in range(1, number_of_days):
    accumilated_reward_ShrinkingCentral[i, 0] = accumilated_reward_ShrinkingCentral[i - 1, 0] + \
                                                rewards_all_daily_ShrinkingCentral[i, 0]

rewards = (list(chain.from_iterable(rewards_all_daily_ShrinkingCentral)))

sns_plot = sns.tsplot(data=np.cumsum(rewards), err_style="boot_traces", n_boot=500)
fig = sns_plot.get_figure()
fig.savefig('accumilated_reward.png')

