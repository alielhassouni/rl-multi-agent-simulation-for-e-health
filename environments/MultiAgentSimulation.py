#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from agents.CommunicationAgent import *
from algorithm.lspi_python.lspi.sample import Sample
from algorithm.lspi_python.lspi.policy import Policy
import numpy as np
import pandas as pd
from algorithm.lspi_python.lspi.basis_functions import OneDimensionalPolynomialBasis, ExactBasis
from algorithm.lspi_python.lspi.lspi import *
from algorithm.lspi_python.lspi.solvers import *
from algorithm.lspi_python.lspi.lspi import *
from datetime import *
import random
from time import gmtime, strftime
from copy import deepcopy
from algorithm.QLearning.qlearn import *
from sklearn.feature_extraction import FeatureHasher
from sklearn.cluster import AgglomerativeClustering
from datetime import datetime
from algorithm.clustering.Distance import LB_Keogh
from algorithm.clustering.KMedoids import KMedoids
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from pprint import pprint
import gc
import hdbscan
import collections
from collections import defaultdict
from scipy.spatial import distance


class MultiAgentSimulation:

    def __init__(self,
                 agent_klass,
                 agent_params,
                 learner_klass,
                 learner_params,
                 agent_profiles_params,
                 number_days_warm_up=None,
                 number_days_learn=None,
                 number_days_apply=None,
                 profile_type = None,
                 number_agents=None,
                 time_granularity=None,
                 results_path=None,
                 db_connection=None,
                 db_name=None
                 ):

        self.agent_klass = agent_klass
        self.agent_params = agent_params
        self.learner_klass = learner_klass
        self.learner_params = learner_params
        self.agent_profiles_params = agent_profiles_params
        self.number_days_warm_up = number_days_warm_up
        self.number_days_learn = number_days_learn
        self.number_days_apply = number_days_apply
        self.profile_type = profile_type
        self.number_agents = number_agents
        self.agents = []
        self.communication_agent = None
        self.time_granularity = time_granularity
        self.counter = 0
        self.current_day_number = 0
        self.current_time = 0
        self.NUMBER_SECONDS_DAY = 86400
        self.NUMBER_SECONDS_HOUR = 60 * 60
        self.results_path = results_path
        self.system_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.warm_up_phase = True
        self.db = db_connection[db_name]
        self.number_clusters = None
        self.current_learning_rate_q_learning = 0.2
        self.learning_rate_reduce_rate = 0.99

    def activate_agents(self):
        """
        Activates agents after the launch of the simulation.
        """
        for i in range(1, self.number_agents+1):
            agent_profile, \
            acceptance, \
            fatigue_after_day, \
            probability_second_workout, \
            agent_type = self.sample_agent_profile((i+1) % 3)

            agent = self.agent_klass(i,
                                     self.agent_params,
                                     self.time_granularity,
                                     agent_profile,
                                     self.agent_profiles_params
                                     )

            agent.agent_acceptance_threshold = acceptance
            agent.fatigue_after_day = fatigue_after_day
            agent.probability_second_workout = probability_second_workout
            agent.agent_type = agent_type
            cluster_id = self.get_cluster_id_agent(agent)
            agent.set_cluster_id(cluster_id)
            agent.update_state(self.current_time, self.db)
            agent.update_planned_activity()
            agent.write_agents_to_database(self.db)
            self.agents.append(agent)

        print(str(self.number_agents) + " agents have been activated...")
        self.number_clusters = 3

    def random_init_fatigue(self):
        """
        Randomly initialises fatigue values.
        """
        for i in range(0, self.number_agents):
            self.agents[i].fatigue = random.SystemRandom().choice([0, 1, 2])

    def get_cluster_id_agent(self, agent):
        """
        Returns initialisation cluster id for agents.
        """
        if self.learner_params["Clustering"] == 'Separate':
            return agent.agent_id

        elif self.learner_params["Clustering"] == 'Pooled':
            return 1

        elif self.learner_params["Clustering"] == 'Grouped':
            return self.get_random_cluster_agent()

    def set_number_clusters(self):
        """
        Sets the global variable for the number of clusters.
        """
        if self.learner_params["Clustering"] == 'Separate':
            self.number_clusters = self.number_agents

        elif self.learner_params["Clustering"] == 'Pooled':
            self.number_clusters = 1

        elif self.learner_params["Clustering"] == 'Grouped':
            self.number_clusters = random.SystemRandom().choice([3, 4, 5, 6, 7, 8, 9])

    def get_random_cluster_agent(self):
        """
        Returns a random cluster id for agents.
        """
        return random.SystemRandom().choice([1, 2, 3, 4, 5, 6, 7])

    def sample_agent_profile(self, type):
        """
        Gets a list of randomly sampled profile types of the agents.
        :param type: agent type.
        :return: a list of randomly sampled profile types of the agents
        """
        activity_profile = None
        planning_profiles = None

        if self.profile_type == 1:
            if type == 0:  # Worker
                activity_profile = "type_1_tight_schedule"
                planning_profiles = "type_1_chronic_planner"
                acceptance_probability = np.random.normal(0.5, 0.01, 1)
                fatigue_after_days = 3
                probability_second_workout = np.random.normal(0.05, 0.005, 1)
                agent_type = "Worker"

            elif type == 1:  # Arnold
                activity_profile = "type_2_normal_schedule"
                planning_profiles = "type_3_mixed_planner_and_spontaneous"
                acceptance_probability = np.random.normal(0.9, 0.01, 1)
                fatigue_after_days = 5
                probability_second_workout = np.random.normal(0.5, 0.01, 1)
                agent_type = "Arnold"

            elif type == 2:  # Retired
                activity_profile = "type_3_light_schedule"
                planning_profiles = "type_2_spontaneous"
                acceptance_probability = np.random.normal(0.7, 0.01, 1)
                fatigue_after_days = 2
                probability_second_workout = np.random.normal(0.05, 0.01, 1)
                agent_type = "Retired"

        elif self.profile_type == 2:
            if type == 0:  # new Worker
                activity_profile = "type_1_tight_schedule"
                planning_profiles = "type_1_chronic_planner"
                acceptance_probability = 0.4
                fatigue_after_days = 2
                probability_second_workout = 0.1
                agent_type = "newWorker"

            elif type == 1:  # new Arnold
                activity_profile = "type_2_normal_schedule"
                planning_profiles = "type_3_mixed_planner_and_spontaneous"
                acceptance_probability = 0.8
                fatigue_after_days = 4
                probability_second_workout = 0.5
                agent_type = "newArnold"

            elif type == 2:  # Arnold
                activity_profile = "type_3_light_schedule"
                planning_profiles = "type_2_spontaneous"
                acceptance_probability = 0.8
                fatigue_after_days = 4
                probability_second_workout = 0.5
                agent_type = "Arnold"

        return [[activity_profile, planning_profiles], acceptance_probability,
                fatigue_after_days, probability_second_workout, agent_type]

    def update_daily_plan_agents(self):
        """
        Updates the planned activity of an agent by calling the function update_planned_activity().
        """
        for i in range(0, self.number_agents):
            self.agents[i].update_planned_activity()

    def clear_intervention_queue_agents(self):
        """
        Clears the intervention queue of an agent.
        """
        for i in range(0, self.number_agents):
            self.agents[i].clear_intervention_queue_agent()

    def update_state_agents(self):
        """
        Updates the state of agents by calling the function update_state()
        from within the agent class.
        """
        for i in range(0, self.number_agents):
            self.agents[i].update_state(self.current_time, self.db)

    def update_current_time(self, steps):
        """
        Updates the current time of the simulation with a number of steps 'steps'.
        """
        self.current_time = self.current_time + steps * self.time_granularity

    def print_activities(self):
        """
        Prints Activities.
        """
        for i in range(0, self.number_agents):
            self.agents[i].print_activities()

    def print_total_rewards(self):
        """
        Prints total rewards.
        """
        print("Total rewards.")
        for i in range(0, self.number_agents):
            print(self.agents[i].reward.get_total_reward())

    def save_current_state_agents(self):
        """
        Saves the current state of agents.
        """
        for i in range(0, self.number_agents):
            state = self.agents[i].get_current_state_agent()
            self.agents[i].state_history.append(state)

    def update_reward_value_agents(self):
        """
        Updates the reward value for agents.
        """
        for i in range(0, self.number_agents):
            self.agents[i].update_reward_value_agent()

    def learn_q(self, q, sample):
        """
        Q-learning with one sample.
        """
        q.learn(hash(tuple(sample[0])), int(sample[1]), float(sample[2]), hash(tuple(sample[3])))
        #print("size Q", str(len(q.q)))
        return q

    def write_LSPI_policy_to_file(self, policies):
        """
        Write the policy to file. Will probably need a day to finish for the sake of understanding and analyzing
        the behaviour of the learned policy.
        :param policies: list of LSPI policies.
        :return:
        """
        i = 0
        for policy in policies.values():
            i=i+1
            textFile = open(self.results_path + self.system_time + str(i) + 'policy.txt', 'w')
            for day in range(0, 7):
                for hour in range(0, 24):
                    for Worked_out_today in range(0, 2):
                        for fatigue in range(0, 8):
                            #for act1 in range(0,2):
                                #for act2 in range(0, 2):
                                    #for act3 in range(0, 2):
                                     #   for act4 in range(0, 2):
                                      #      for act5 in range(0, 2):
                                       #         for act6 in range(0, 2):
                                                        best_action = policy.select_action(
                                                        self.get_feature_approximation_state([7, 24, 2, 8],
                                                        np.array([day, hour, Worked_out_today, fatigue])))
                                                        textFile.write(
                                                        "Day: %d\r\n "
                                                        "Hour: %d\r\n "
                                                        "Worked_out_today: %d\r\n "
                                                        "fatigue: %d\r\n "
                                                        "best_action: %d\r\n" %
                                                        (day, hour, Worked_out_today, fatigue, best_action))
            textFile.close()

    def write_reward_to_db(self, db):
        """
        Writes rewards to database.
        """
        for i in range(0, self.number_agents):
            self.agents[i].write_reward_to_db(db)
            self.agents[i].init_reward()

    def write_state_to_db(self, db):
        """
        Writes states to database.
        """
        for i in range(0, self.number_agents):
            self.agents[i].write_state_to_db(db, self.agents[i].get_current_state_agent())
            self.agents[i].init_state()

    def write_action_to_db(self, db):
        """
        Writes actions to database.
        """
        for i in range(0, self.number_agents):
            self.agents[i].write_action_to_db(db)

    def reset_hourly_intervention_received_indicator(self):
        """
        Resets the boolean variables worked out today or went to sleep today to their default value: False.
        """
        for i in range(0, self.number_agents):
            self.agents[i].reset_occurrence_target_activities_current_hour()

    def init_activities_performed_last_hour(self):
        """
        Initializes the activities array performed last hour.
        """
        for i in range(0, self.number_agents):
            self.agents[i].init_activities_performed_last_hour()

    def cluster_2(self):
        if self.learner_params["Clustering"] == 'GroupedBench':
            for i in range(0, self.number_agents):
                agent = self.agents[i]
                if agent.agent_profile[0] == "type_1_tight_schedule":
                    self.agents[i].cluster_id = 1
                elif agent.agent_profile[0] == "type_2_normal_schedule":
                    self.agents[i].cluster_id = 2
                elif agent.agent_profile[0] == "type_3_light_schedule":
                    self.agents[i].cluster_id = 3

                self.agents[i].update_cluster_id(self.db, self.agents[i].cluster_id)

            self.number_clusters = 3

    def cluster(self):
        """
        Cluster agents based on their traces.
        """
        if self.learner_params["Clustering"] == 'Grouped':
            if self.learner_params["Cluster_type"] == 'KMedoids':
                traces = None
                scheduling_profile = None

                if self.learner_params["Features"] == 'Normal':
                    # Clustering using the 11 standard features (reward, day of week, hour of the day, etc.)
                    traces, scheduling_profile = self.read_clustering_data()
                elif self.learner_params["Features"] == 'Advanced':
                    # Clustering using the derived features
                    traces, scheduling_profile = self.read_generated_clustering_data()

                dtw_days_matching_of_profiles = self.get_sorted_average_amount_activity_per_day_per_profile(
                    self.agent_profiles_params)
                distances = self.pre_calculate_distances(traces,
                                                         scheduling_profile,
                                                         dtw_days_matching_of_profiles,
                                                         norm=False)

                K_Medoids = KMedoids()
                best_k = 0
                best_score = -1000000000
                best_clusters = None

                for k in range(2, min(6, self.number_agents - 1)):
                    clusters, curr_medoids = K_Medoids.cluster(distances=distances, k=k)
                    silhouette_avg = silhouette_score(distances, clusters, metric="precomputed")
                    print(clusters)
                    print("__________________________________________________________________________")
                    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
                    print("__________________________________________________________________________")

                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_clusters = clusters
                        best_k = k
                        print("__________________________________________________________________________")
                        print("Best K =", best_k, "The best average silhouette_score is :", best_score)
                        print("__________________________________________________________________________")

                print(best_clusters)
                self.number_clusters = best_k
                self.clusters = best_clusters
                self.assign_clusters_to_agents()

            elif self.learner_params["Cluster_type"] == 'AgglomerativeClustering':
                traces = None
                scheduling_profile = None
                if self.learner_params["Features"] == 'Normal':
                    # Clustering using the 11 standard features (reward, day of week, hour of the day, etc.)
                    traces, scheduling_profile = self.read_clustering_data()
                elif self.learner_params["Features"] == 'Advanced':
                    # Clustering using the derived features
                    traces, scheduling_profile = self.read_generated_clustering_data()
                dtw_days_matching_of_profiles = self.get_sorted_average_amount_activity_per_day_per_profile(
                    self.agent_profiles_params)
                # Clustering using hard clustering and precomputed distances
                distances = self.pre_calculate_distances(traces,
                                                         scheduling_profile,
                                                         dtw_days_matching_of_profiles,
                                                         norm=False)
                best_k = 0
                best_score = -1000000000
                best_clusters = None

                for k in range(2, min(7, self.number_agents - 1)):  # Add paramters in config
                    clusters = AgglomerativeClustering(k, affinity='precomputed', linkage='complete').fit_predict(distances)
                    silhouette_avg = silhouette_score(distances, clusters, metric="precomputed")
                    print(clusters)
                    print("__________________________________________________________________________")
                    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
                    print("__________________________________________________________________________")

                    if silhouette_avg > best_score:
                        best_score = silhouette_avg
                        best_clusters = clusters
                        best_k = k
                        print("__________________________________________________________________________")
                        print("Best K =", best_k, "The best average silhouette_score is :", best_score)
                        print("__________________________________________________________________________")

                print(best_clusters)
                self.number_clusters = best_k
                self.clusters = best_clusters
                self.assign_clusters_to_agents_hdbscan()

            elif self.learner_params["Cluster_type"] == 'HDBScan':
                traces = None
                scheduling_profile = None

                if self.learner_params["Features"] == 'Normal':
                    # Clustering using the 11 standard features (reward, day of week, hour of the day, etc.)
                    traces, scheduling_profile = self.read_clustering_data()
                elif self.learner_params["Features"] == 'Advanced':
                    # Clustering using the derived features
                    traces, scheduling_profile = self.read_generated_clustering_data()

                dtw_days_matching_of_profiles = self.get_sorted_average_amount_activity_per_day_per_profile(
                    self.agent_profiles_params)

                #clustering using hard clustering and precomputed distances
#                distances = self.pre_calculate_distances(traces,
#                                                         scheduling_profile,
#                                                         dtw_days_matching_of_profiles,
#                                                         norm=False)
#                cluster_labels = hdbscan.HDBSCAN(min_cluster_size=5, metric='precomputed').fit_predict(distances)
#                most_common = collections.Counter(cluster_labels).most_common(1)[0][0]
#                for i in range(0, len(cluster_labels)):
#                    if cluster_labels[i] == -1:
#                        cluster_labels[i] = most_common

                #clustering with soft clustering to deal with outliers (can't use precomputed distances with this method)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data ='true', metric='euclidean').fit(traces)
                soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
                cluster_labels = [np.argmax(x) for x in soft_clusters]
                print("CLUSTER LABELS"+str(cluster_labels))
                self.number_clusters = len(set(cluster_labels))
                self.clusters = cluster_labels
                self.assign_clusters_to_agents_hdbscan()

    def assign_clusters_to_agents(self):
        """
        Assign cluster id to agents.
        """
        cluster_set = sorted(list(set(self.clusters)))
        for i in range(0, self.number_agents):
            self.agents[i].cluster_id = cluster_set.index(self.clusters[i]) + 1
            self.agents[i].update_cluster_id(self.db, self.agents[i].cluster_id)

    def assign_clusters_to_agents_hdbscan(self):
        """
        Assign cluster id to agents.
        """
        for i in range(0, self.number_agents):
            self.agents[i].cluster_id = self.clusters[i] + 1
            self.agents[i].update_cluster_id(self.db, self.agents[i].cluster_id)

    def pre_calculate_distances(self, traces, scheduling_profile, dtw_days_matching_of_profiles, norm=False):
        """
        Pre-calculate the distance matrix for clustering.
        """
        if self.learner_params["Distance"] == 'dtw':
            LBKeogh = LB_Keogh()
            distances = np.zeros(shape=(len(traces), len(traces)))
            for i in range(0, len(traces)):
                for j in range(0, len(traces)):
                    if i == j:
                        distances[i, j] = 0
                    elif i < j:
                        dist = LBKeogh.calculate_distance(traces[i], traces[j], scheduling_profile[i], scheduling_profile[j],
                                                       dtw_days_matching_of_profiles)
                        #dist = LBKeogh.LB_Keogh(s1=traces[i], s2=traces[j], r=1)
                        distances[i, j] = dist
                        distances[j, i] = dist
            if norm:
                distances = normalize(distances, axis=1, norm='l1')
            return distances

        elif self.learner_params["Distance"] == 'Euclidean':
            distances = np.zeros(shape=(len(traces), len(traces)))

            for i in range(0, len(traces)):
                for j in range(0, len(traces)):
                    if i == j:
                        distances[i, j] = 0
                        print(i, j)
                    elif i < j:
                        dist = distance.euclidean(traces[i], traces[j])
                        distances[i, j] = dist
                        distances[j, i] = dist
            if norm:
                distances = normalize(distances, axis=1, norm='l1')
            return distances

    def read_clustering_data(self):
        """
        Read clustering data from database.
        """
        query = {'state_reward_trace': {'$exists': 1}}
        projection = {'_id': 0, 'state_reward_trace': 1}
        sort = [('agent_id',1)]
        query_result = self.db.agent_state.find(query, projection, sort=sort)
        result = np.zeros(shape=(query_result.count(), int((len(query_result[0]["state_reward_trace"]))/11*11)))

        for i in range(0, query_result.count()):
            y = 0
            for e in range(0, len(query_result[i]["state_reward_trace"])):
                # Reward
                if np.mod(e+1,11) == 1:
                    result[i][y]= float(query_result[i]["state_reward_trace"][e])
                    y= y+1

                # Day of the week
                elif np.mod(e+1,11) == 2:
                    result[i][y]= float(query_result[i]["state_reward_trace"][e])/7
                    y= y+1

                # Hour of the day
                elif np.mod(e+1, 11) == 3:
                    result[i][y]= float(query_result[i]["state_reward_trace"][e])/24
                    y=y+1

                # Worked out today
                elif np.mod(e+1,11) == 4:
                    result[i][y]= float(query_result[i]["state_reward_trace"][e])
                    y=y+1

                # Fatigue
                elif np.mod(e+1,11) == 5:
                    result[i][y]= float(query_result[i]["state_reward_trace"][e])/8
                    y=y+1

                # Activities
                elif np.mod(e+1,11) == 6 \
                        or np.mod(e+1,11) == 7 \
                        or np.mod(e+1,11) == 8 \
                        or np.mod(e+1,11) == 9 \
                        or np.mod(e+1,11) == 10 \
                        or np.mod(e+1,11) == 0:
                    result[i][y]= float(query_result[i]["state_reward_trace"][e])
                    y=y+1

        query = {'scheduling_profile': {'$exists': 1}}
        projection = {'_id': 0, 'scheduling_profile': 1}
        query_result_2 = self.db.agent_state.find(query, projection)
        return result, query_result_2

    def read_generated_clustering_data(self):
        """
        Read the advance clustering data from database.
        """
        query = {'activities_log': {'$exists': 1}}
        projection = {'_id': 0, 'activities_log': 1}
        sort = [('agent_id',1)]
        query_result = self.db.agent_state.find(query, projection, sort=sort)

        durations = [defaultdict(list) for x in range(query_result.count())]

        # The array number of columns is 7 because the agent can perform a maximum of 6 activities + IDLE
        results = np.zeros(shape=(query_result.count(), 6))
        total_time = self.current_day_number * 86400

        for i in range(0, query_result.count()):
            r = query_result[i]["activities_log"]

            # Looping over the query result of one agent
            # Index 0 is the name of the activity, index 3 is the duration of said activity
            for e in range(0, len(r)):
                durations[i][r[e][0]].append(r[e][3])

        for i in range(0, len(results)):
            results[i][0]=sum(durations[i]["breakfast"])/total_time
            results[i][1]=sum(durations[i]["work"])/total_time
            results[i][2]=sum(durations[i]["lunch"])/total_time
            results[i][3]=sum(durations[i]["dinner"])/total_time
            results[i][4]=sum(durations[i]["sleep"])/total_time
            results[i][5]=sum(durations[i]["workout"])/total_time
#           results[i][6]=1 - sum(results[i])
            print(results[i])

        query = {'scheduling_profile': {'$exists': 1}}
        projection = {'_id': 0, 'scheduling_profile': 1}
        query_result_2 = self.db.agent_state.find(query, projection)
        return results, query_result_2

    def get_sorted_average_amount_activity_per_day_per_profile(self, agent_profiles_params):
        """
        Prepare activity vector per day per profile type.
        :param agent_profiles_params: agent profile parameters.
        :return: the sorted average amount of activity per day per profile.
        """
        result = dict()

        for profile in agent_profiles_params["activity_profiles"].keys():
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for activity in agent_profiles_params["activity_profiles"][profile].keys():
                b = agent_profiles_params["activity_profiles"]\
                    [profile][activity]["probability_performing_activity_per_weekday"]
                a = np.add(a, b)
            a = np.argsort(a)
            result.update({profile: a[::-1]})
        return result

    def Q_learn(self, update=False, old_Q_learned_policy = None):
        """
        Perform Q-Learning.
        :param update: Boolean for updating or learning policy from scratch.
        :param old_Q_learned_policy: the old learned policy is update is True.
        :return: a dict of policies.
        """
        policy_dict = dict()
        number_clusters = self.number_clusters

        if not update:
            for i in range(1, number_clusters + 1):
                data_agent = self.read_data_cluster(i)
                samples = self.convert_data_for_Q_learning(data_agent,
                                                           update=False,
                                                           experience_replay=False)
                Q = QLearn([0, 1],
                           epsilon=0.05,
                           alpha=self.current_learning_rate_q_learning,
                           gamma=0.95)

                for sample in samples:
                    Q.learn(hash(tuple(sample[0])), int(sample[1]), float(sample[2]), hash(tuple(sample[3])))

                policy_dict.update({i: deepcopy(Q)})
        else:
            for i in range(1, number_clusters + 1):
                data_agent = self.read_data_cluster(i)
                samples = self.convert_data_for_Q_learning(data_agent,
                                                           update=True,
                                                           experience_replay=True)
                Q = deepcopy(old_Q_learned_policy.get(i))

                for sample in samples:
                    Q.learn(hash(tuple(sample[0])), int(sample[1]), float(sample[2]), hash(tuple(sample[3])))

                policy_dict.update({i:deepcopy(Q)})
        return deepcopy(policy_dict)

    def convert_data_for_Q_learning(self, data, update, experience_replay):
        """
        Prepares data for Q-Learning.
        :param data: data.
        :param update: Boolean for updating or learning policy from scratch.
        :param experience_replay: Experience replay parameter boolean.
        :return: Samples for Q-learning
        """
        samples = []
        length = data.count()

        if update:
            start = len(data[0]['states'])-1
        else:
            start = 0

        if experience_replay:
            exp_replay = 250
        else:
            exp_replay = 1

        if length > 0:
            for i in range(0, length):
                states = data[i]['states']
                actions = data[i]['actions']
                rewards = data[i]['rewards']
                number_samples = len(states)
                for j in range(max(start-exp_replay, 0), number_samples-1):
                    sample = []
                    sample.insert(0, self.convert_state_to_list(states[j]))
                    sample.insert(1, actions[j])
                    sample.insert(2, np.float(rewards[j]))
                    sample.insert(3, self.convert_state_to_list(states[j+1]))
                    samples.append(sample)
        return samples

    def learn_lspi(self, update=False):
        """
        Learn an LSPI policy
        :param update: Boolean for updating or learning policy from scratch.
        :return: a dict of policies.
        """
        policy_dict = dict()
        number_clusters = self.number_clusters

        if update:
            for i in range(1, number_clusters + 1):
                data_agent = self.read_data_cluster(i)
                samples = self.convert_data_to_LSPI_sample_update(data_agent)
                basis = ExactBasis(np.array([2]*53), 2)
                policy = Policy(basis=basis,
                                discount=0.95,
                                explore=0.01,
                                tie_breaking_strategy=Policy.TieBreakingStrategy.FirstWins)
                solver = LSTDQSolver()
                print("Updating LSPI policy for cluster: " + str(i) + " out of: " + str(self.number_clusters))
                p = learn(samples,
                          policy,
                          solver,
                          max_iterations=20,
                          epsilon=0.00001)
                policy_dict.update({i: p})
        else:
            for i in range(1, number_clusters + 1):
                data_agent = self.read_data_cluster(i)
                samples = self.convert_data_to_LSPI_sample(data_agent)
                basis = ExactBasis(np.array([2]*53), 2)
                print("Learning LSPI policy for cluster: " + str(i) + " out of: " + str(self.number_clusters))
                policy = Policy(basis=basis,
                                discount=0.95,
                                explore=0.01,
                                tie_breaking_strategy=Policy.TieBreakingStrategy.FirstWins)
                solver = LSTDQSolver()
                p = learn(samples,
                          policy,
                          solver,
                          max_iterations=20,
                          epsilon=0.00001)
                policy_dict.update({i:p})
        return policy_dict

    def read_data_cluster(self, cluster_id):
        """
        Read data from db.
        :param cluster_id: cluster id.
        :return: query result.
        """
        query_result = self.db.agent_state.find({"cluster_id": cluster_id}, {"states": 1, "rewards": 1, "actions": 1, "_id": 0})
        return query_result

    def get_feature_approximation_state(self, n_features, state):
        """
        Return a one-hot-encoded representation of the state of the agent.
        Returns basis function features for LSPI. [7, 24, 2, 8, 2, 2 , 2, 2, 2, 2]
        :param n_features: the number of features in the state representation.
        :param state: The state vector.
        :return: A transformed representation of the state.
        """
        result = np.zeros(sum(n_features), dtype=int)

        for i in range(0, len(n_features)):
            first_position = int(np.sum(n_features[0:i]))
            result[first_position + int(state[i])] = 1
        return result

    def convert_data_to_LSPI_sample_update(self, data):
        """
        Converts data to LSPI sample objects for the update period.
        :param data: The data.
        :return: The converted samples.
        """
        samples = []
        length = data.count()

        if length > 0:
            for i in range(0, length):
                states = data[i]['states']
                actions = data[i]['actions']
                rewards = data[i]['rewards']
                number_samples = len(states)

                for j in range(self.number_days_warm_up*24, number_samples - 1):
                    sample = Sample(
                        self.get_feature_approximation_state([7, 24, 2, 8, 2, 2, 2, 2, 2, 2],
                                                             np.array(self.convert_state_to_list(
                                                                 states[j]))),
                        actions[j],
                        np.float(rewards[j]),
                        self.get_feature_approximation_state([7, 24, 2, 8, 2, 2, 2, 2, 2, 2],
                                                             np.array(self.convert_state_to_list(
                                                                 states[j + 1]))),
                        absorb=False
                    )
                    samples.append(sample)
        return samples

    def convert_data_to_LSPI_sample(self, data):
        """
        Converts data to LSPI sample objects.
        :param data: The data.
        :return: The converted samples.
        """
        samples = []
        length = data.count()

        if length > 0:
            for i in range(0, length):
                states = data[i]['states']
                actions = data[i]['actions']
                rewards = data[i]['rewards']
                number_samples = len(states)

                for j in range(0, number_samples-1):
                    sample = Sample(
                        self.get_feature_approximation_state([7, 24, 2, 8, 2, 2, 2, 2, 2, 2],
                                                             np.array(self.convert_state_to_list(
                                                                 states[j]))),
                        actions[j],
                        np.float(rewards[j]),
                        self.get_feature_approximation_state([7, 24, 2, 8, 2, 2, 2, 2, 2, 2],
                                                             np.array(self.convert_state_to_list(
                                                                 states[j+1]))),
                        absorb=False)
                    samples.append(sample)
        return samples

    def convert_state_to_list(self, state):
        """
        Converts a state array to a list.
        :param state: state.
        :return: list of state features.
        """
        a = state
        c = list()
        fatigue = a["fatigue"]
        worked_out_today = a["worked_out_today"]
        day_of_week = a["day_of_week"]
        hour_of_day = a["hour_of_day"]
        activities_performed = list(a["activities_performed"])
        c.append(day_of_week) # from 1 to 7
        c.append(hour_of_day) # from 0 to 23
        c.append(worked_out_today) # 0 or 1
        c.append(fatigue) # from 0 to 7
        c = c + activities_performed[0:6] # each one can be 0 or 1
        return c

    def simulate(self, algorithm):
        """
        Run simulator with algorithms: Q or LSPI.
        :param algorithm: "Q-learning" or "LSPI".
        """
        warm_up = True
        learn_now = False
        end_condition = self.number_days_warm_up

        while self.current_day_number < end_condition:
            self.update_current_time(1)
            self.counter = self.counter + 1 * self.time_granularity
            self.print_activities()

            if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                self.reset_hourly_intervention_received_indicator()

            if self.counter % self.NUMBER_SECONDS_HOUR == self.NUMBER_SECONDS_HOUR - 1:
                self.write_action_to_db(self.db)
                pass

            if self.current_time % self.NUMBER_SECONDS_DAY == 0:
                self.current_day_number = self.current_day_number + 1
                print("Current day number: " + str(self.current_day_number))
                self.clear_intervention_queue_agents()
                self.update_daily_plan_agents()
                self.current_learning_rate_q_learning = self.current_learning_rate_q_learning * self.learning_rate_reduce_rate

            if algorithm == "LSPI":
                if self.current_day_number == self.number_days_warm_up and \
                                        self.current_time % self.NUMBER_SECONDS_DAY == 0:
                    warm_up = False
                    learn_now = True
                    end_condition = self.number_days_warm_up + self.number_days_learn
                    self.cluster()
                    self.cluster_2()
                    policy = self.learn_lspi(update=False)

                if self.current_day_number == self.number_days_warm_up + self.number_days_learn and \
                                        self.current_time % self.NUMBER_SECONDS_DAY == 0:
                    warm_up = False
                    learn_now = False
                    apply_now = True
                    end_condition = self.number_days_warm_up + self.number_days_learn + self.number_days_apply
                    # Batch learn LSPI
                    policy = self.learn_lspi(update=True)

                if warm_up:
                    if self.current_time % self.NUMBER_SECONDS_DAY == 0:
                        self.communication_agent.handle_communication(self.current_time,
                                                                      self.current_day_number,
                                                                      random_policy=True,
                                                                      policy_object=None)

                    if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                        self.write_reward_to_db(self.db)
                        self.write_state_to_db(self.db)
                        pass

                elif learn_now:
                    if self.current_time % self.NUMBER_SECONDS_DAY == 0:
                        policy = self.learn_lspi(update=False)

                    if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                        self.reset_hourly_intervention_received_indicator()
                        # Sum up rewards last hour, save it to db and clear array
                        self.write_reward_to_db(self.db)
                        self.write_state_to_db(self.db)
                        self.communication_agent.handle_communication(self.current_time,
                                                                      self.current_day_number,
                                                                      random_policy=False,
                                                                      policy_object=deepcopy(policy))

                elif apply_now:
                    if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                        self.reset_hourly_intervention_received_indicator()
                        self.write_reward_to_db(self.db)
                        self.write_state_to_db(self.db)
                        # Sum up rewards last hour, save it to db and clear array
                        self.communication_agent.handle_communication(self.current_time,
                                                                      self.current_day_number,
                                                                      random_policy=False,
                                                                      policy_object=deepcopy(policy))

            elif algorithm == "Q":
                if self.current_day_number == self.number_days_warm_up and \
                                        self.current_time % self.NUMBER_SECONDS_DAY == 0:
                    warm_up = False
                    learn_now = True
                    end_condition = self.number_days_warm_up + self.number_days_learn
                    self.cluster()
                    self.cluster_2()
                    policy = self.Q_learn(update=False)

                if self.current_day_number == self.number_days_warm_up + self.number_days_learn and \
                                        self.current_time % self.NUMBER_SECONDS_DAY == 0:
                    warm_up = False
                    learn_now = False
                    apply_now = True
                    end_condition = self.number_days_warm_up + self.number_days_learn + self.number_days_apply

                if warm_up:
                    if self.current_time % self.NUMBER_SECONDS_DAY == 0:
                        print("Q-learning: Sending interventions to all agents using random policy _________")
                        self.communication_agent.handle_communication(self.current_time,
                                                                      self.current_day_number,
                                                                      random_policy=True,
                                                                      policy_object=None)
                    if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                        self.write_reward_to_db(self.db)
                        self.write_state_to_db(self.db)

                elif learn_now:
                    if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                        self.write_reward_to_db(self.db)
                        self.write_state_to_db(self.db)

                        policy = self.Q_learn(update=True,
                                              olQ_learnd_policy=deepcopy(policy))

                        self.communication_agent.handle_communication(self.current_time,
                                                                      self.current_day_number,
                                                                      random_policy=False,
                                                                      policy_object=deepcopy(policy))

                elif apply_now:
                    if self.counter % self.NUMBER_SECONDS_HOUR == 0:
                        self.write_reward_to_db(self.db)
                        self.write_state_to_db(self.db)
                        self.communication_agent.handle_communication(self.current_time,
                                                                      self.current_day_number,
                                                                      random_policy=False,
                                                                      policy_object=deepcopy(policy))
            else:
                exit(1)

            self.update_state_agents()
            self.update_reward_value_agents()

    def run(self):
        """
        Run the Simulation.
        """
        startTime = datetime.now()
        self.db.agent_state.drop()
        print("The database: " + str(self.db.agent_state) + " was dropped")
        self.db.create_collection(name="agent_state")
        print("The collection: agent_state was created...")
        self.activate_agents()
        self.set_number_clusters()
        self.update_daily_plan_agents()
        self.random_init_fatigue()
        self.communication_agent = CommunicationAgent(0, self, self.learner_params["Algorithm"], self.db)
        self.counter = 0
        self.simulate(self.learner_params["Algorithm"])
        print(datetime.now()-startTime)
