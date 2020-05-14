from pymongo import MongoClient
from collections import defaultdict
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
import numpy as np

db_connection = MongoClient('localhost', 27017)

#####Change name according to collection you want to analyse#####
db_name = 'experiment_103_LSPI_Normal_Grouped_KMedoids_DTW'

db = db_connection[db_name]
print(db)

profile1 = 'Worker'
profile2 = 'Arnold'
profile3 = 'Retired'


def agent_type(agent):
    if(agent['planning_profile'] == 'type_2_spontaneous'): #Retired
        return profile3
    elif(agent['planning_profile'] == 'type_1_chronic_planner'): #Worker
        return profile1
    else: #Arnold
        return profile2


#setting query
query = {'cluster_id':  {'$exists': 1}, 'scheduling_profile': {'$exists': 1}, 'planning_profile': {'$exists': 1}}
projection = {'_id': 0, 'cluster_id': 1, 'scheduling_profile': 1, 'planning_profile': 1}
sort = [('agent_id',1)]
inv_sort = [('agent_id', -1)]

query_result = db.agent_state.find(query, projection, sort=sort, limit=50)
query_result2 = db.agent_state.find(query, projection, sort=inv_sort, limit=50)
print("here")
print(query_result)
cluster_index = []
cluster_index2 = []

for result in query_result:
    cluster_index.append(result)

for result in query_result2:
    cluster_index2.append(result)

cluster_index2 = cluster_index2[::-1]
#creating sorted list of clustering indexes with profile info
cluster_index = cluster_index+cluster_index2


cluster_list = [[] for _ in range(7)]


####Populating clusters####
for e in cluster_index:
    if e['cluster_id'] == 1:
        cluster_list[0].append(agent_type(e))
    elif e['cluster_id'] == 2:
        cluster_list[1].append(agent_type(e))
    elif e['cluster_id'] == 3:
        cluster_list[2].append(agent_type(e))
    elif e['cluster_id'] == 4:
        cluster_list[3].append(agent_type(e))
    elif e['cluster_id'] == 5:
        cluster_list[4].append(agent_type(e))
    elif e['cluster_id'] == 6:
        cluster_list[5].append(agent_type(e))
    else:
        cluster_list[6].append(agent_type(e))


def cluster_distr(cluster):
    d = defaultdict(int)
    total = len(cluster)
    if total == 0:
        return 'Empty'
    else:
        for e in cluster:
            d[e] += 1
        p1p = d[profile1]
        p2p = d[profile2]
        p3p = d[profile3]
        return profile1, profile2, profile3, p1p,p2p,p3p

####LIST OF ELEMENTS IN CLUSTERS####
clusters = [[] for _ in range(len(cluster_list))]

####POPULATING LIST OF ELEMENTS IN CLUSTERS####
for i in range(0, len(cluster_list)):
    clusters[i].append(cluster_distr(cluster_list[i]))

print(clusters)

####QUERING FOR REWARDS####
query = {'rewards':  {'$exists': 1}}
projection = {'_id': 0, 'rewards': 1}
sort = [('agent_id',1)]
inv_sort = [('agent_id', -1)]

query_list = db.agent_state.find(query, projection, sort=sort, limit=50)
query_list2 = db.agent_state.find(query, projection, sort=inv_sort, limit=50)

reward_list = []
reward_list2 = []

for result in query_list:
    reward_list.append(result)

for result in query_list2:
    reward_list2.append(result)

reward_list2 = reward_list2[::-1]
#creating sorted list of reward values
reward_list = reward_list+reward_list2


#####FUNCTION TO CALCULATION AVERAGE DAILY REWARD OF AN AGENT#########
def average_daily_reward(agent):
    average_reward = 0
    reward_list = []
    reward = 0
    days = 0
    for i in range(0, len(agent['rewards'])):
        if i % 24 == 23:
            reward += float(agent['rewards'][i])
            reward_list.append(reward)
            reward = 0
            days+=1
        else:
            reward += float(agent['rewards'][i])
    average_reward = sum(reward_list) / days
    return average_reward

####AVERAGE DAILY REWARD FOR ALL AGENTS####
def average_daily_reward_all(reward_list):
    list_of_rewards = []
    for e in reward_list:
        list_of_rewards.append(average_daily_reward(e))
    return sum(list_of_rewards)/100




####CALCULATE AVERAGE REWARD PER DAY FOR ONE AGENT###
def reward_per_day(agent):
    reward_per_day = []
    sum = 0
    for i in range(0, len(agent['rewards'])):
        if i % 24 == 23:
            sum += float(agent['rewards'][i])
            reward_per_day.append(sum)
            sum = 0
        else:
            sum += float(agent['rewards'][i])
    return reward_per_day

####CALCULATE CUMULATIVE DAILY REWARD OVER ALL AGENTS####
def sum_reward(reward_list):
    n_agents = int(len(reward_list))
    n_days = int(len(reward_list[0]['rewards'])/24)
    reward_collection = np.zeros(shape=(n_agents, n_days))
    name = db_name + '.csv'
    for i in range(0, n_agents):
        reward_collection[i] = reward_per_day(reward_list[i])
    sum_array = reward_collection.sum(axis=0)
    np.savetxt(name, sum_array, delimiter=",")
    return sum_array



#sum_reward(reward_list)




####LIST OF REWARDS#####
#rewards = []

####POPULATING LIST OF REWARDS####
#for i in range(0, len(reward_list)):
#    rewards.append(average_daily_reward(reward_list[i]))



####PLOTTING THE CLUSTER###

'''
x = np.random.randn(10)
y = np.random.randn(10)
Cluster = np.array([0, 1, 1, 1, 3, 2, 2, 3, 0, 2])    # Labels of cluster 0 to 3
centers = np.random.randn(4, 2)

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,c=Cluster,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.savefig('clustering.png', dpi=fig.dpi)
'''

#def plot_cluster(clusters):
#    for e in clusters:
 #       if e[0] == 'Empty':
  #           pass
   #     else:
    #        e[0][3]