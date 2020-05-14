#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from Activity import *
from random import *
from queue import *
from reinforcement_learning.Reward import *
from reinforcement_learning.State import *
from reinforcement_learning.Action import *
import time
import math
from copy import deepcopy
import numpy as np
import pandas as pd
from datetime import *


class Agent:

    def __init__(self, agent_id, params, time_granularity, agent_profile, agent_profiles_params):
        self.agent_id = agent_id
        self.agent_type = "Typeless"
        self.cluster_id = None
        self.params = params
        self.agent_profile = agent_profile
        self.agent_profiles_params = agent_profiles_params
        self.activities_list = self.params['activities']
        self.activities = {}
        self.agent_acceptance_threshold = None
        self.ACTIVITY_QUEUE_MAX_SIZE = 1000000
        self.current_activity = None
        self.current_activity_end_time = None
        self.current_state = 'IDLE'
        self.current_time = 0
        self.current_time_ymdhms = 0
        self.INTERVENTION_QUEUE_MAX_SIZE = 100
        self.interventions = Queue(maxsize=self.INTERVENTION_QUEUE_MAX_SIZE)
        self.lined_up_intervention = None
        self.next_intervention = None
        self.NUMBER_SECONDS_DAY = 86400
        self.NUMBER_SECONDS_HOUR = 60 * 60
        self.second_activity = None
        self.second_activity_end_time = None
        self.states = self.params['states']
        self.time_granularity = time_granularity
        self.worked_out_today = False
        self.worked_out_today_twice = False
        self.went_to_sleep_today = False
        self.received_intervention_current_hour = 0
        self.reward = Reward(self.agent_id)
        self.state_history = []
        self.fatigue = SystemRandom().choice([0, 1, 2, 3, 4, 5, 6, 7])
        self.activities_performed_last_hour = None
        self.init_activities_performed_last_hour()
        self.pending_activity_params = None
        self.fatigue_after_day = 2
        self.probability_second_workout = None

    def init_reward(self):
        """
        Initialize reward object for this agent.
        """
        self.reward = Reward(self.agent_id)

    def init_state(self):
        """
        Initialize the activity state of this agent.
        """
        self.init_activities_performed_last_hour()

    def init_activities_performed_last_hour(self):
        """
        Initialize the activity state vector for this agent by setting all activities performed to zero.
        """
        self.activities_performed_last_hour = \
            {
                "breakfast": 0,
                "work": 0,
                "lunch": 0,
                "workout": 0,
                "dinner": 0,
                "sleep": 0
            }

    def get_current_state_agent(self):
        """
        Get the current state of this agent.
        :return: Return a state object with the state values of this agent.
        """
        state = State(self.agent_id,
                      self.current_time_ymdhms,
                      self.time_granularity,
                      self.agent_acceptance_threshold,
                      self.fatigue)

        state.append_state(self.current_state == 'ACTIVE',
                           self.current_state == 'IDLE',
                           self.worked_out_today,
                           self.went_to_sleep_today,
                           self.current_activity,
                           self.current_activity_end_time,
                           self.current_time,
                           self.current_time_ymdhms,
                           self.second_activity,
                           self.second_activity_end_time,
                           self.fatigue)
        return state

    def write_agents_to_database(self, db):
        """
        Append agent to database.
        :param db: A mongodb connection.
        """
        db.agent_state.insert_one(
            {
                "agent_id": self.agent_id,
                "cluster_id": int(self.cluster_id),
                "scheduling_profile": str(self.agent_profile[0]),
                "planning_profile": str(self.agent_profile[1]),
                "rewards": [],
                "states": [],
                "actions": [],
                "state_reward_trace": [],
                "activities_log": []
            }
        )

    def update_cluster_id(self, db, cluster_id):
        """
        Update the cluster id of agent.
        :param db: A mongodb connection.
        :param cluster_id: A numerical cluster id.
        """
        db.agent_state.update({"agent_id": self.agent_id}, {"$set": {"cluster_id": int(cluster_id)}})

    def write_reward_to_db(self, db):
        """
        Write current scalar reward to database.
        :param db: A mongodb connection.
        """
        reward = self.reward.get_total_reward()
        db.agent_state.update({"agent_id": self.agent_id}, {"$push": {"rewards": str(reward)}})
        db.agent_state.update({"agent_id": self.agent_id}, {"$push": {"state_reward_trace": str(reward)}})

    def write_action_to_db(self, db):
        """
        Write most recent reward to database.
        :param db: A mongodb connection.
        """
        action = self.received_intervention_current_hour
        db.agent_state.update({"agent_id": self.agent_id}, {"$push": {"actions": action}})

    def write_state_to_db(self, db, state):
        """
        Write the current state of the agent to the database.
        :param db: A mongodb connection.
        :param state: The current state of the agent.
        """
        activities_performed = []
        for key, value in self.activities_performed_last_hour.items():
            activities_performed.append(value)

        if self.current_time == self.NUMBER_SECONDS_HOUR:
            db.agent_state.update(
                {"agent_id": self.agent_id},
                {"$push": {"states": {"day_of_week": None,
                                      "hour_of_day": None,
                                      "worked_out_today": None,
                                      "fatigue": None,
                                      "activities_performed": None}
                           }
                 }
            )

        else:
            db.agent_state.update(
                {"agent_id": self.agent_id},
                {"$push": {"states":
                               {"day_of_week": datetime.strptime(state.current_time_ymdhms,
                                                                 "%Y-%m-%d %H:%M:%S").isoweekday(),
                                "hour_of_day": datetime.strptime(state.current_time_ymdhms, "%Y-%m-%d %H:%M:%S").hour,
                                "worked_out_today": int(state.worked_out_today),
                                "fatigue": state.fatigue,
                                "activities_performed": activities_performed
                                }
                           }
                 }
            )
            db.agent_state.update(
                {"agent_id": self.agent_id},
                {"$push": {"state_reward_trace":
                               datetime.strptime(state.current_time_ymdhms, "%Y-%m-%d %H:%M:%S").isoweekday()
                           }
                 }
            )
            db.agent_state.update(
                {"agent_id": self.agent_id},
                {"$push": {"state_reward_trace":
                               datetime.strptime(state.current_time_ymdhms, "%Y-%m-%d %H:%M:%S").hour
                           }
                 }
            )
            db.agent_state.update(
                {"agent_id": self.agent_id},
                {"$push": {"state_reward_trace":
                               int(state.worked_out_today)
                           }
                 }
            )
            db.agent_state.update(
                {"agent_id": self.agent_id},
                {"$push": {"state_reward_trace":
                               min(state.fatigue, 7)  # Limit fatigue range to 7 for LSPI
                           }
                 }
            )

            for key, value in self.activities_performed_last_hour.items():
                db.agent_state.update(
                    {"agent_id": self.agent_id},
                    {"$push": {"state_reward_trace": value}
                     }
                )

    def write_activities_log_to_db(self, db, first_activity=True):
        """
        Write a log of all activities to db.
        :param db: A mongodb connection.
        :param first_activity: Boolean indicating if the user is performing multiple activities at the same time.
        """
        if first_activity:
            log_array = [self.current_activity.label,
                         datetime.utcfromtimestamp(self.current_activity.start_time_activity).strftime(
                             "%Y-%m-%d %H:%M:%S"),
                         datetime.utcfromtimestamp(self.current_activity_end_time).strftime("%Y-%m-%d %H:%M:%S"),
                         self.current_activity_end_time - self.current_activity.start_time_activity]

            db.agent_state.update({"agent_id": self.agent_id}, {"$push": {"activities_log": log_array}})
        else:
            log_array = [self.second_activity.label,
                         datetime.utcfromtimestamp(self.second_activity.start_time_activity).strftime(
                             "%Y-%m-%d %H:%M:%S"),
                         datetime.utcfromtimestamp(self.second_activity_end_time).strftime("%Y-%m-%d %H:%M:%S"),
                         self.second_activity_end_time - self.second_activity.start_time_activity]
            db.agent_state.update({"agent_id": self.agent_id}, {"$push": {"activities_log": log_array}})

    def get_current_state_for_lspi_policy(self):
        """
        Get the current state represenation suitable for the LSPI algorithm.
        :return: the state of the agent.
        """
        # Monday is 0 and Sunday is 6.
        weekday = \
            datetime.strptime(self.current_time_ymdhms,
                              "%Y-%m-%d %H:%M:%S").isoweekday()
        # Hour of day
        hour_of_day = \
            datetime.strptime(self.current_time_ymdhms,
                              "%Y-%m-%d %H:%M:%S").hour
        # Worked out today
        new_state_worked_out_today = int(self.worked_out_today)

        # fatigue
        fatigue = min(self.fatigue, 7)

        return self.get_feature_approximation_state(
            [7, 24, 2, 8, 2, 2, 2, 2, 2, 2],
            self.convert_state_to_list(
                [weekday,
                 hour_of_day,
                 new_state_worked_out_today,
                 fatigue,
                 self.get_performed_activities_last_hour()]))

    def convert_state_to_list(self, state):
        """
        Convert the state to a list.
        :param state: the current state vector.
        :return: a list of state features.
        """
        a = list(state)
        b = a[4]
        c = a[0:4]
        c = c + b[0:6]
        return c

    def get_feature_approximation_state(self, n_features, state):
        """
        Return a one-hot-encoded representation of the state of the agent.
        Returns basis function features for LSPI.
        :param n_features: the number of features in the state representation.
        :param state: The state vector.
        :return: A transformed representation of the state.
        """
        result = np.zeros(sum(n_features), dtype=int)
        for i in range(0, len(n_features)):
            first_position = int(np.sum(n_features[0:i]))
            result[first_position + int(state[i])] = 1
        return result

    def get_performed_activities_last_hour(self):
        """
        Returned a list that indicates whether an activity was performed during the last hour.
        :return: A list of acitivties with value indicating performed last hour or not.
        """
        activities_performed = []
        for key, value in self.activities_performed_last_hour.items():
            activities_performed.append(value)

        return activities_performed

    def set_cluster_id(self, cluster_id):
        """
        Set the cluster id of the agent.
        :param cluster_id: integer cluster id of the agent.
        """
        self.cluster_id = cluster_id

    def get_current_state_for_policy(self):
        """
        Return the state representation of the agent for policy inference.
        :return: A list of state features.
        """
        # Monday is 0 and Sunday is 6.
        weekday = \
            datetime.strptime(self.current_time_ymdhms,
                              "%Y-%m-%d %H:%M:%S").isoweekday()
        # Hour of day
        hour_of_day = \
            datetime.strptime(self.current_time_ymdhms,
                              "%Y-%m-%d %H:%M:%S").hour
        # Worked out today
        new_state_worked_out_today = int(self.worked_out_today)

        # fatigue
        fatigue = min(7, self.fatigue)
        return [weekday, hour_of_day, new_state_worked_out_today, fatigue]

    def update_planned_activity(self):
        """
        Update the state of the users based on the planned activities sampled from schedule.
        """
        self.update_fatigue_variable()
        schedule_profile = str(self.agent_profile[0])
        all_activities = self.agent_profiles_params["activity_profiles"][schedule_profile].keys()

        self.activities = {}

        for activity in all_activities:
            activity_params = self.agent_profiles_params["activity_profiles"][schedule_profile][activity]

            parameters = {
                "label": activity,
                "start_time": round(randint(
                    activity_params["starting_time_hour_range"][0] * self.NUMBER_SECONDS_HOUR,
                    activity_params["starting_time_hour_range"][1] * self.NUMBER_SECONDS_HOUR) / self.time_granularity),
                "duration": round(randint(
                    activity_params["duration_hour_range"][0] * self.NUMBER_SECONDS_HOUR,
                    activity_params["duration_hour_range"][1] * self.NUMBER_SECONDS_HOUR) / self.time_granularity),
                "standard_deviation": round(randint(
                    activity_params["standard_deviation_seconds"][0],
                    activity_params["standard_deviation_seconds"][0]) / self.time_granularity),
                "prio": activity_params["priority_activities"],
                "probability_performing_activity_per_weekday":
                    activity_params["probability_performing_activity_per_weekday"]
            }

            weekday = datetime.strptime(str(self.current_time_ymdhms), "%Y-%m-%d %H:%M:%S").isoweekday()

            if parameters["probability_performing_activity_per_weekday"][weekday - 1] > 0:
                self.activities[activity] = parameters

        self.reset_occurrence_target_activities_current_day()

    def get_time_of_day_ymdhms(self):
        """
        Get the current time for this agent.
        :return: current time of agent in format: Year:Month:Day:Hour:Minute:Second.
        """
        return self.current_time_ymdhms

    def update_time_of_day_ymdhms(self):
        """
        Update time of the day in format: Year:Month:Day:Hour:Minute:Second.
        """
        self.current_time_ymdhms = \
            datetime.utcfromtimestamp(self.current_time).strftime("%Y-%m-%d %H:%M:%S")

    def clear_intervention_queue_agent(self):
        """
        Clear the intervention queue object of agent by initializing a new Queue object.
        """
        self.interventions = Queue(maxsize=self.INTERVENTION_QUEUE_MAX_SIZE)

    def set_current_activity_end_time(self, start_time, duration):
        """
        Update the end time of the current activity by adding the duration to the start_time.
        :param start_time: The start time of the activity.
        :param duration: The duration of the activity.
        """
        self.current_activity_end_time = start_time + duration

    def update_state(self, current_time, database):
        """
        Update the current state of the agent.
        Update the time-related components first, the update the states of the agent given current state.
        Possible states: ['ACTIVE', 'IDLE'].
        ACTIVE: The agent is performing a certain activities from the possible activities in
        the activities dictionary: activities.
        IDLE: The agent is not performing any of the activities from the possible activities in
        the activities dictionary: activities.
        :param current_time: The current time for this agent.
        :param database: A mongodb connection
        """
        self.current_time = current_time
        self.update_time_of_day_ymdhms()

        if self.current_state == 'IDLE':
            self.update_state_idle()
        elif self.current_state == 'ACTIVE':
            self.update_state_active(database)

        self.check_interventions_queue()
        self.send_next_intervention()
        self.register_occurrence_target_activities_current_day()

        if self.current_state == 'ACTIVE':
            self.activities_performed_last_hour[self.current_activity.label] = 1
            if self.second_activity is not None:
                self.activities_performed_last_hour[self.second_activity.label] = 1

    def reset_occurrence_target_activities_current_hour(self):
        """
        Set the variable received_intervention_current_hour to zero.
        """
        self.received_intervention_current_hour = 0

    def register_occurrence_target_activities_current_day(self):
        """
        Check if agent performed activity from target activities and flag variables with True if activity is found.
        """
        if not self.current_activity == None:
            if self.current_activity.label == "workout":
                self.worked_out_today = True
            elif self.current_activity.label == "sleep_":
                self.went_to_sleep_today = True

    def reset_occurrence_target_activities_current_day(self):
        """
        Reset activity performed variable by setting them to False.
        """
        self.worked_out_today = False
        self.worked_out_today_twice = False
        self.went_to_sleep_today = False

    def update_fatigue_variable(self):
        """
        Update the level of fatigue of the agent.
        """
        if self.worked_out_today:
            self.fatigue += 1
            self.fatigue = min(7, self.fatigue)
        elif self.worked_out_today_twice:
            self.fatigue += 2
            self.fatigue = min(7, self.fatigue)
        else:
            self.fatigue = 0

    def get_samples_Q_learning(self):
        """
        Get the previous state representation along with action, reward and next state suitable for Q-learning.
        :return: the state of the agent.
        """
        # state1, action1, reward, state2
        new_state = self.get_current_state_agent()

        new_state_weekday = \
            datetime.strptime(new_state.current_time_ymdhms,
                              "%Y-%m-%d %H:%M:%S").isoweekday()

        new_state_hour_of_day = \
            datetime.strptime(new_state.current_time_ymdhms,
                              "%Y-%m-%d %H:%M:%S").hour

        new_state_worked_out_today = int(new_state.worked_out_today)

        # fatigue
        new_state_fatigue = new_state.fatigue

        activities = self.get_performed_activities_last_hour()

        state1 = self.convert_state_to_list([int(new_state_weekday),
                                             int(new_state_hour_of_day),
                                             int(new_state_worked_out_today),
                                             min(np.int(new_state_fatigue), 7),
                                             activities])

        sample = []
        sample.insert(0, state1)
        return sample

    def check_interventions_queue(self):
        """
        Check if there is an intervention lined up.
        If not, get next intervention an put it as lined up.
        If an intervention is lined up, check if time has arrived.
        Check if activity already occurred.
        If no, check if agent responds to the intervention.
        If agent responds, update the time of the corresponding activity.
        """
        if self.lined_up_intervention is None:
            if self.interventions.qsize() > 0:
                self.lined_up_intervention = self.interventions.get()

    def update_reward_value_agent(self):
        """
        Update the reward value for agent.
        """
        if self.current_activity is not None:
            if self.current_activity.label == "workout":
                if self.current_time == self.current_activity.end_time_activity:
                    self.reward.append(self.get_reward_value("suggested_activity_completed", None))
                elif self.current_time < self.current_activity.end_time_activity:
                    self.reward.append(self.get_reward_value("suggested_activity_performed_for_one_tick", None))

        if self.current_time % self.NUMBER_SECONDS_HOUR == 0:
            self.reward.append(self.get_reward_value("fatigue", None))

    def get_reward_value(self, stage, activity):
        """
        :param stage: At what stage are we now. Proposing activity, performing activity and fatigue management.
        :param activity: What is the current activity.
        :return: reward
        """
        reward = 0.0
        if stage == "suggestion_accepted":
            reward = reward + 1
        elif stage == "suggestion_rejected":
            reward = reward + -1
        elif stage == "suggested_activity_completed":
            reward = reward + 10.0
        elif stage == "suggested_activity_performed_for_one_tick":
            reward = reward + 0.01
        elif stage == "fatigue":
            if self.fatigue >= self.fatigue_after_day:
                reward = reward + min(7, self.fatigue) * 0.1 * -1
        return reward

    def send_next_intervention(self):
        """
        Send next intervention in line.
            """
        if self.lined_up_intervention is not None:
            if self.lined_up_intervention.created_at < self.current_time:
                self.lined_up_intervention = None

        if self.lined_up_intervention is not None:
            if self.lined_up_intervention.created_at == self.current_time:
                self.received_intervention_current_hour = 1
                schedule_profile = str(self.agent_profile[0])
                activity_id = self.lined_up_intervention.activity_id
                activity_params = self.agent_profiles_params["activity_profiles"][schedule_profile][activity_id]
                activity = Activity(label=activity_id,
                                    duration_mean=round(randint(
                                        activity_params["duration_hour_range"][0] * self.NUMBER_SECONDS_HOUR,
                                        activity_params["duration_hour_range"][1] * self.NUMBER_SECONDS_HOUR) /
                                                        self.time_granularity),
                                    duration_sd=round(activity_params["standard_deviation_seconds"][0] /
                                                      self.time_granularity),
                                    start_time_activity=self.current_time,
                                    end_time_activity=None,
                                    intervention=True,
                                    fatigue=self.fatigue)

                intervention_accepted = self.suggest_intervention(self.lined_up_intervention, activity)

                if intervention_accepted:
                    self.schedule_intervention_activity(self.lined_up_intervention, activity)
                    self.reward.append(
                        self.get_reward_value("suggestion_accepted", activity))
                else:
                    self.lined_up_intervention = None
                    self.reward.append(self.get_reward_value("suggestion_rejected", None))
            else:
                self.reward.append(0)
        else:
            self.reward.append(0)

    def schedule_intervention_activity(self, lined_up_intervention, activity):
        """
        Schedule intervention acitivity if lined up.
        :param lined_up_intervention: an intervention object.
        :param activity: the activity to be scheduled.
        """
        print(self.current_state)
        if self.current_state == "IDLE":
            self.set_new_activity(activity)
            self.remove_planned_activity(lined_up_intervention.activity_id)

        elif self.current_state == "ACTIVE":
            if self.current_activity.label != activity.label:
                if self.second_activity is None:

                    if self.current_time < self.current_activity_end_time:
                        if activity is not None:
                            self.remove_planned_activity(lined_up_intervention.activity_id)
                            self.update_state_new_activity(activity)
                    elif self.current_activity_end_time == self.current_time:
                        self.current_state = 'IDLE'
                        self.current_activity_end_time = None

                        if activity is not None:
                            self.remove_planned_activity(lined_up_intervention.activity_id)
                            self.set_new_activity(activity)
                else:

                    if self.current_time < self.current_activity_end_time:
                        if self.current_time < self.second_activity_end_time:
                            pass
                        elif self.second_activity_end_time == self.current_time:
                            self.second_activity_end_time = None
                            self.second_activity = None
                    elif self.current_activity_end_time == self.current_time:
                        self.current_activity = None
                        self.current_activity_end_time = None

                        if self.current_time < self.second_activity_end_time:
                            self.current_activity_end_time = self.second_activity_end_time
                            self.current_activity = self.second_activity
                            self.second_activity = None
                            self.second_activity_end_time = None
                        elif self.second_activity_end_time == self.current_time:
                            self.second_activity_end_time = None
                            self.second_activity = None
                            self.current_state = 'IDLE'

        self.lined_up_intervention = None

    def remove_planned_activity(self, activity_id):
        """
        Remove a planned activity from schedule.
        :param activity_id: the id of the acitivity.
        """
        self.activities.pop(activity_id, None)

    def suggest_intervention(self, intervention, activity):
        """
        Accept intervention if time to first possible gap for intervention is within preference range of agent.
        Accept intervention if agent performs activity during a certain day (schedule).
        If activity is already scheduled then just look at the horizon.
        If already worked out today then decline.
        """
        planning_profile = str(self.agent_profile[1])

        preferred_planning_horizon_min_in_hours = \
            self.agent_profiles_params["planning_profiles"][planning_profile]["preferred_planning_horizon_in_hours_min"]

        preferred_planning_horizon_mean_in_hours = \
            self.agent_profiles_params["planning_profiles"][planning_profile][
                "preferred_planning_horizon_in_hours_mean"]

        preferred_planning_horizon_sd_in_hours = \
            self.agent_profiles_params["planning_profiles"][planning_profile]["standard_deviation_in_hours"]

        min = ((preferred_planning_horizon_min_in_hours * self.NUMBER_SECONDS_HOUR) / self.time_granularity)
        mean = ((preferred_planning_horizon_mean_in_hours * self.NUMBER_SECONDS_HOUR) / self.time_granularity)
        stdev = ((preferred_planning_horizon_sd_in_hours * self.NUMBER_SECONDS_HOUR) / self.time_granularity)
        max = np.random.normal(mean, stdev) + min

        activity_already_planned = self.activity_planned_starting_time(intervention.activity_id)
        accept = False

        print("Acceptance function entered>>>")
        if not self.worked_out_today or (
                random() >= (1 - self.probability_second_workout) and not self.worked_out_today_twice):
            if self.has_gap_in_preferred_planning_horizon(intervention.activity_id, activity, min, max):
                if self.agent_type == "Worker":
                    if self.current_state == "IDLE":
                        print("Agent type WORKER accepted intervention.")
                        accept = True
                        if activity_already_planned:
                            self.remove_planned_activity(intervention.activity_id)
                    elif not self.second_activity == None:
                        if self.second_activity.label == "lunch":
                            print("Agent type WORKER accepted intervention.")
                            accept = True
                            if activity_already_planned:
                                self.remove_planned_activity(intervention.activity_id)
                elif self.agent_type == "Arnold":
                    print("Agent type ARNOLD accepted intervention.")
                    accept = True
                    if activity_already_planned:
                        self.remove_planned_activity(intervention.activity_id)
                elif self.agent_type == "Retired":
                    if self.current_state == "IDLE":
                        print("Agent type RETIRED accepted intervention.")
                        accept = True
                        if activity_already_planned:
                            self.remove_planned_activity(intervention.activity_id)
        else:
            accept = False
        return accept

    def activity_planned_starting_time(self, activity_id):
        """
        Check if activity is already planned and return planned activity object with starting time.
        :param activity_id: the id of the activity to be checked.
        :return: planned activity object with starting time.
        """
        if activity_id in self.activities.keys():
            activity_already_planned = True
            return activity_already_planned

    def has_gap_in_preferred_planning_horizon(self, activity_id, activity, min, max):
        """
        Checked is agent has gap in planning horizon that fits length of the acitivity.
        :param activity_id: the id of the activity to be checked.
        :param activity: the object of the activity to be checked.
        :param min: minimum starting time.
        :param max: maximum starting time.
        :return: boolean value.
        """
        activities = deepcopy(self.activities)
        activities.pop(activity_id, None)
        time_of_day = (self.current_time % 86400) / self.time_granularity
        min = ((self.current_time % 86400) / self.time_granularity) + min
        max = ((self.current_time % 86400) / self.time_granularity) + max

        cols = {'Activity_ID': [],
                'Starting_time': [],
                'Ending_time': []}

        activity_df = pd.DataFrame(columns=cols)

        for key in activities:
            temp_df = pd.DataFrame(columns=cols)
            temp_df['Activity_ID'] = [activities[key]['label']]
            temp_df['Starting_time'] = [activities[key]['start_time']]
            temp_df['Duration'] = [activities[key]['duration']]
            activity_df = activity_df.append([temp_df])

        activity_df = activity_df.sort_values('Starting_time', ascending=True)
        activity_df = activity_df[activity_df['Starting_time'] >= time_of_day]

        if self.current_state == 'ACTIVE':
            time_of_day = self.current_activity_end_time

        for i in range(0, len(activity_df)):
            if activity_df.iloc[i]['Starting_time'] < (time_of_day):
                pass
            else:
                time_diff = activity_df.iloc[i]['Starting_time'] - (time_of_day)
                if time_diff >= activity.duration:
                    if time_of_day >= min and time_of_day <= max:
                        return True
                    else:
                        time_of_day = activity_df.iloc[i]['Starting_time'] + activity_df.iloc[i]['Duration']
                else:
                    time_of_day = activity_df.iloc[i]['Starting_time'] + activity_df.iloc[i]['Duration']

            if time_of_day > max:
                return False
        return False

    def time_to_first_gap_in_schedule(self, activity_id, activity):
        """
        Find the amount of time for the first gap in schedule that fits activity.
        :param activity_id: the id of the activity to be checked.
        :param activity: the object of the activity to be checked.
        :return: return starting time of gap.
        """
        activities = deepcopy(self.activities)
        activities.pop(activity_id, None)

        time_of_day = (self.current_time % 86400) / self.time_granularity

        cols = {'Activity_ID': [],
                'Starting_time': [],
                'Ending_time': []}
        activity_df = pd.DataFrame(columns=cols)

        for key in activities:
            temp_df = pd.DataFrame(columns=cols)
            temp_df['Activity_ID'] = [activities[key]['start_time']]
            temp_df['Starting_time'] = [activities[key]['start_time']]
            temp_df['Duration'] = [activities[key]['duration']]
            activity_df = activity_df.append([temp_df])

        activity_df = activity_df.sort_values('Starting_time', ascending=True)

        if self.current_state == 'ACTIVE':
            time_of_day = self.current_activity_end_time

        for i in range(0, len(activity_df)):

            if activity_df.iloc[i]['Starting_time'] - time_of_day >= activity.duration:
                return (time_of_day - self.current_time) / self.time_granularity

            else:
                time_of_day = activity_df.iloc[i]['Starting_time'] + activity_df.iloc[i]['Duration']

        return 10000000000000

    def update_state_idle(self):
        """
        Update the state of the agent given that the current state of the agent is: 'IDLE'.
        Check if there is any planned activity that should be started.
        Update current activity if a new activity has arrived.
        """
        activity = self.get_planned_activity()
        if activity is not None:
            self.set_new_activity(activity)

    def update_state_active(self, database):
        """
        Update the state of the agent given that the current state of the agent is: 'ACTIVE'.
        Check if there is any planned activity that should be started.
        Update current activity/activities if a new activity has arrived.
        """
        if self.second_activity is None:
            self.update_state_active_first_activity(database)
        else:
            self.update_state_active_second_activity(database)

    def update_state_active_first_activity(self, database):
        """
        Update the state of the agent given that the current state of the agent is:
        'ACTIVE' and agent is performing just 1 activity.
        Check if there is any planned activity that should be started.
        Update current activity if a new activity has arrived.
        """
        activity = self.get_planned_activity()
        if activity is not None:
            print(activity.label, self.agent_type)
        if self.current_time < self.current_activity_end_time:
            if activity is not None:
                self.update_state_new_activity(activity)
        elif self.current_activity_end_time == self.current_time:
            self.write_activities_log_to_db(database)
            self.current_state = 'IDLE'
            self.current_activity_end_time = None
            if activity is not None:
                self.set_new_activity(activity)
        # self.log_activity()

    def update_state_active_second_activity(self, database):
        """
        Update the state of the agent given that the current state of the agent is:
        'ACTIVE' and agent is performing 2 activities.
        Check if there is any planned activity that should be started.
        Update current activities if a new activity has arrived.
        """
        if self.current_time < self.current_activity_end_time:
            if self.current_time < self.second_activity_end_time:
                pass
            elif self.second_activity_end_time == self.current_time:
                self.write_activities_log_to_db(database, first_activity=False)
                self.second_activity_end_time = None
                self.second_activity = None

        elif self.current_activity_end_time == self.current_time:
            self.write_activities_log_to_db(database, first_activity=False)
            self.current_activity = None
            self.current_activity_end_time = None

            if self.current_time < self.second_activity_end_time:
                self.current_activity_end_time = self.second_activity_end_time
                self.current_activity = self.second_activity
                self.second_activity = None
                self.second_activity_end_time = None
            elif self.second_activity_end_time == self.current_time:
                self.second_activity_end_time = None
                self.second_activity = None
                self.current_state = 'IDLE'

    def print_activities(self):
        """
        Print the activities given the current state of the agent.
        """
        if self.current_state == 'ACTIVE':
            if self.second_activity is None:
                print(self.agent_id,
                      self.agent_type,
                      self.current_time_ymdhms,
                      self.current_state,
                      self.current_time,
                      "First_activity",
                      self.current_activity.label)
            else:
                print(self.agent_id,
                      self.agent_type,
                      self.current_time_ymdhms,
                      self.current_state,
                      self.current_time,
                      "Second_activity",
                      self.current_activity.label,
                      self.second_activity.label)
        else:
            print(self.agent_id,
                  self.agent_type,
                  self.current_time_ymdhms,
                  self.current_state,
                  self.current_time,
                  'No Activity')

    def set_new_activity(self, activity):
        """
        Set new activity global variables.
        activity: an activity object.
        """
        self.current_state = 'ACTIVE'
        self.current_activity = activity
        self.current_activity_end_time = activity.end_time_activity

    def set_second_activity(self, activity):
        """
        Set new second activity global variables.
        activity: an activity object.
        """
        self.current_state = 'ACTIVE'
        self.second_activity = activity
        self.second_activity_end_time = activity.end_time_activity

    def update_state_new_activity(self, activity):
        """
        Update the state of the agent given that the current state is:
        'ACTIVE' and agent is already performing 1 activity.
        Allowed second activity is currently lunch (e.g. lunch during work lunch break).
        """
        for key in self.activities:
            if self.current_activity.label == key:
                if activity.label not in self.activities[key]['prio']:
                    if activity.label != "lunch":
                        self.activities[activity.label]['start_time'] = self.current_activity_end_time + 1
                    elif activity.label == "lunch":
                        self.second_activity = activity
                        self.second_activity_end_time = activity.end_time_activity
                else:
                    self.current_activity_end_time = activity.end_time_activity
                    self.current_activity = activity

    def get_planned_activity(self):
        """
        Check if there is any planned activity arriving at a certain moment in time.
        Return the activity object if an activity occurs.
        """
        time_of_day = (self.current_time % 86400) / self.time_granularity
        activity = None

        for key in self.activities:
            if self.activities[key]['start_time'] == time_of_day:
                activity = Activity(self.activities[key]['label'], self.activities[key]['duration']
                                    / self.time_granularity,
                                    self.activities[key]['standard_deviation']
                                    / self.time_granularity, self.current_time, 0)
                if activity.label == "sleep":
                    self.register_occurrence_sleep_activities_current_day()
                return activity
        return activity

    def register_occurrence_sleep_activities_current_day(self):
        """
        Check if agent performed activity from target activities and flag variables with True.
        """
        if not self.current_activity == None:
            if self.current_activity.label == "workout":
                if self.worked_out_today:
                    self.worked_out_today_twice = True
                else:
                    self.worked_out_today = True
            elif self.current_activity.label == "sleep_":
                self.went_to_sleep_today = True

    def log_activity(self):
        """
        Log current activity of the agent.
        Distinguish between the states ACTIVE and IDLE.
        """
        if self.current_state == 'IDLE':
            self.log_activity_idle()
        elif self.current_state == 'ACTIVE':
            self.log_activity_active()

    def log_activity_active(self):
        """
        Log current activity of the agent if active.
        Distinguish between one and two activities.
        """
        self.activity_log[self.current_activity.label].append([self.current_activity.label,
                                                               self.current_time_ymdhms])
        if self.second_activity is not None:
            self.activity_log[self.current_activity.label].append([self.second_activity.label,
                                                                   self.current_time_ymdhms])

    def log_activity_idle(self):
        """
        Log current activity of the agent if idle.
        Distinguish between one and two activities.
        """
        self.activity_log['NO_ACTIVITY'].append(1)

    def plan_second_activity(self, activity):
        """
        Update the state of the agent given that the current state is:
        'ACTIVE' and agent is already performing 1 activity.
        """
        self.activities.put(activity)
        self.second_activity_end_time = activity.end_time_activity
        self.current_activity = activity
        self.current_state = 'ACTIVE'
        self.current_activity = activity
