#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

import numpy as np
import time


class State:

    def __init__(self,
                 agent_id,
                 created_time_stamp,
                 time_granularity,
                 acceptance_rate_agent,
                 fatigue):

        self.agent_id = agent_id
        self.created_time_stamp = created_time_stamp
        self.time_granularity = time_granularity
        self.accepetance_rate_agent = acceptance_rate_agent
        self.active = None
        self.idle = None
        self.worked_out_today = None
        self.went_to_sleep_today = None
        self.current_activity = None
        self.current_activity_end_time = None
        self.current_time = 0
        self.current_time_ymdhms = 0
        self.second_activity = None
        self.second_activity_end_time = None
        self.current_activity_time_left = None
        self.fatigue = 0

    def append_state(self, active, idle, worked_out_today, went_to_sleep_today, current_activity,
                     current_activity_end_time, current_time, current_time_ymdhms, second_activity,
                     second_activity_end_time, fatigue):
        """
        Append state values.
        :param active: Boolean value.
        :param idle: Boolean value.
        :param worked_out_today: Boolean value.
        :param went_to_sleep_today: Boolean value.
        :param current_activity: categorical.
        :param current_activity_end_time: numeric.
        :param current_time: numeric.
        :param current_time_ymdhms: timestamp.
        :param second_activity: categorical.
        :param second_activity_end_time: numeric.
        :param fatigue: numerical.

        """
        self.active = active
        self.ilde = idle
        self.worked_out_today = worked_out_today
        self.went_to_sleep_today = went_to_sleep_today
        self.current_activity = current_activity
        self.current_activity_end_time == current_activity_end_time
        self.current_time = current_time
        self.current_time_ymdhms = current_time_ymdhms
        self.second_activity = second_activity
        self.second_activity_end_time = second_activity_end_time
        self.fatigue = fatigue

    def print_state(self):
        """
        Print state values.
        """
        print("Agent Id: " + str(self.agent_id) +
              " created_time_stamp: " + str(self.created_time_stamp) +
              " time_granularity: " + str(self.time_granularity) +
              " accepetance_rate_agent: " + str(self.accepetance_rate_agent) +
              " active: " + str(self.active) +
              " idle: " + str(self.idle) +
              " worked_out_today: " + str(self.worked_out_today) +
              " went_to_sleep_today: " + str(self.went_to_sleep_today) +
              " current_activity: " + str(self.current_activity) +
              " current_activity_end_time: " + str(self.current_activity_end_time) +
              " current_time: " + str(self.current_time) +
              " current_time_ymdhms: " + str(self.current_time_ymdhms) +
              " second_activity: " + str(self.second_activity) +
              " second_activity_end_time: " + str(self.second_activity_end_time) +
              " current_activity_time_left: " + str(self.current_activity_time_left)
              )
        time.sleep(1)
