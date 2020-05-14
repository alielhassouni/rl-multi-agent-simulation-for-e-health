#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from array import *


class Action:

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.actions = array('i', [])

    def append(self, action):
        """
        Append a action to the array of actions.
        :param action: Action value.
        """
        self.actions.append(action)

    def get(self, position):
        """
        Return a action from the array of actions.
        :param position: position of action.
        :return: action value.
        """
        a = self.actions.tolist()
        return a[position]

    def insert(self, position, value):
        """
        Insert a action to the array of actions in a defined position.
        :param position: position of action.
        :param value: action value.
        """
        self.actions.insert(position, value)

    def remove(self, position):
        """
        Remove a action from the array of actions on a defined position.
        :param position: position of action.
        """
        self.actions.remove(self, position)

    def remove_last_action(self):
        """
        Remove the last action from the array of actions.
        """
        self.actions.pop()

    def length(self):
        """
        Return the length of the array of actions.
        :return: Length of the array of actions.
        """
        return self.actions.__len__()

    def count_action(self, value):
        """
        Count actions with value.
        :param value: action value.
        :return: Count.
        """
        return self.actions.count(self, value)

    def get_actions(self):
        """
        Return the array of actions.
        :return: array of actions
        """
        return self.actions

    def get_actions_list(self):
        """
        Return a list of actions.
        :return: a list of actions.
        """
        return self.actions.tolist()

    def get_action_array_info(self):
        """
        Return the action array buffer start address in memory
        and number of elements in the actions array.
        :return: action array info.
        """
        return self.actions.buffer_info()

    def get_total_action(self):
        """
        Return the total sum of actions in the actions array.
        :return: get sum of all action values.
        """
        return sum(self.get_actions_list())

    def get_average_action(self):
        """
        Return the total sum of actions in the actions array.
        :return: get average of all action values.
        """
        return self.get_total_action() / self.actions.__len__()

    def get_last_action(self):
        """
        Return the latest action from the actions array.
        :return: the latest action from the actions array.
        """
        return self.get(len(self.actions) - 1)