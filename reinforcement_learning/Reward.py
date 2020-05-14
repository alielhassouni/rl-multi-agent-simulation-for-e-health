#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from array import *


class Reward:

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.rewards = array('d', [])

    def append(self, reward):
        """
        Append a reward to the array of rewards.
        :param reward: Reward value.
        """
        self.rewards.append(reward)

    def get(self, position):
        """
        Return a reward from the array of rewards.
        :param position: position of reward.
        :return: reward value.
        """
        r = self.rewards.tolist()
        return r[position]

    def insert(self, position, value):
        """
        Insert a reward to the array of rewards in a defined position.
        :param position: position of reward.
        :param value: reward value.
        """
        self.rewards.insert(position, value)

    def remove(self, position):
        """
        Remove a reward from the array of rewards on a defined position.
        :param position: position of reward.
        """
        self.rewards.remove(self, position)

    def remove_last_reward(self):
        """
        Remove the last reward from the array of rewards.
        """
        self.rewards.pop()

    def length(self):
        """
        Return the length of the array of rewards.
        """
        return self.rewards.__len__()

    def count_reward(self, value):
        """
        Return the array of rewards.
        :param value: reward value.
        :return: Count.
        """
        return self.rewards.count(self, value)

    def get_rewards(self):
        """
        Return the array of rewards.
        :return: array of rewards.
        """
        return self.rewards

    def get_rewards_list(self):
        """
        Return a list of rewards.
        :return: a list of rewards..
        """
        return self.rewards.tolist()

    def get_reward_array_info(self):
        """
        Return the reward array buffer start address in memory
        and number of elements in the rewards array.
        :return: reward array info.
        """
        return self.rewards.buffer_info()

    def get_total_reward(self):
        """
        Return the total sum of rewards in the rewards array.
        :return: get sum of all reward values.
        """
        return sum(self.get_rewards_list())

    def get_average_reward(self):
        """
        Return the total sum of rewards in the rewards array.
        :return: get average of all reward values.
        """
        return self.get_total_reward() / self.rewards.__len__()

    def get_last_reward(self):
        """
        Return the latest reward from the rewards array.
        :return: the latest reward from the rewards array.
        """
        return self.get(len(self.rewards)-1)
