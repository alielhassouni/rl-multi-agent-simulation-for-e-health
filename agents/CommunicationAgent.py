#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from queue import *
from reinforcement_learning.DefaultPolicy import *
import numpy as np
from copy import deepcopy
from datetime import datetime


class CommunicationAgent:

    def __init__(self, agent_id, environment, algorithm, db):
        self.environment = environment
        self.agents = environment.agents
        self.agent_id = agent_id
        self.current_time = 0
        self.current_day_number = 0
        self.default_policy = DefaultPolicy()
        self.learned_policy = None
        self.random_policy = True
        self.algorithm = algorithm
        self.db = db

    def handle_communication(self, current_time, current_day_number, random_policy, policy_object):
        """
        Handle communication between algorithm/app and agents.
        This agent (the communication agent has a perfect view on
        the agents and sees all internal states of these agents).
        :param current_time: The current time w.rt. communication agent (i.e. system time).
        :param current_day_number: The number of the current day.
        :param random_policy: A default policy.
        :param policy_object: The learned policy.
        """
        self.current_time = current_time
        self.current_day_number = current_day_number
        self.learned_policy = deepcopy(policy_object)
        self.random_policy = random_policy

        for agent in self.agents:
            for type in ["workout"]:
                self.send_intervention(agent, type)

    def send_intervention(self, agent, type):
        """
        Send an intervention to an agent at a certain time point with a certain message.
        :param agent: agent object.
        :param type: agent type.
        """
        if self.random_policy:
            self.send_intervention_default_policy(agent, type)
        elif not self.random_policy:
            self.send_intervention_learned_policy(agent, type)

    def send_intervention_learned_policy(self, agent, type):
        """
        Send an intervention to an agent at a certain time point with a certain message using a learned policy.
        :param agent: agent object.
        :param type: agent type.
        """
        if self.algorithm == "Q":
            best_action = self.learned_policy.get(agent.cluster_id).chooseAction(
                hash(tuple(agent.get_samples_Q_learning()[0])))
        elif self.algorithm == "LSPI":
            best_action = self.learned_policy.get(agent.cluster_id).select_action(np.array(
                agent.get_current_state_for_lspi_policy()))

        if best_action == 1:
            print("ACTION SENT" + str(self.algorithm))
            intervention = self.default_policy.get_intervention_with_real_policy(type,
                                                                                 self.current_time,
                                                                                 self.current_day_number)
            agent_id = agent.agent_id
            self.agents[agent_id - 1].interventions.put(intervention)

    def send_intervention_default_policy(self, agent, type):
        """
        Send an intervention to an agent at a certain time point with a certain message using a default policy.
        :param agent: agent object.
        :param type: agent type.
        """
        intervention = self.default_policy.get_intervention(type, self.current_time,
                                                            self.current_day_number)
        agent_id = agent.agent_id
        self.agents[agent_id - 1].interventions.put(intervention)

    def print_intervention_data(self, intervention):
        """
        Prints properties from intervention object.
        """
        print("Intervention created:: ")
        print("  ")
        print("Current time: " + str(self.current_time))
        print("Sending time: " + str(intervention.created_at))
        print("Intervention type: " + str(intervention.activity_id))
        print("  ")
