#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from queue import *
from Message import *
from Intervention import *
import random


class DefaultPolicy:

    def __init__(self):
        self.current_time = 0
        self.current_day_number = 0
        self.interventions = Queue()
        self.NUMBER_SECONDS_DAY = 86400
        self.NUMBER_SECONDS_HOUR = 60 * 60
        self.HOUR_TIME_WORKOUT = list(range(9, 21))
        self.HOUR_TIME_SLEEP = list(range(0, 24))

    def get_intervention(self, activity_id, current_time, current_day_number):
        """
        Sends an intervention to an agent at a certain time point with a certain message.
        :param activity_id: activity id.
        :param current_time: the current time.
        :param current_day_number: the current day number
        :return: intervention object.
        """
        self.current_time = current_time
        self.current_day_number = current_day_number

        time_intervention = self.define_time_new_intervention(activity_id, seed=True)
        message_content = self.define_message_new_intervention(activity_id)

        message = Message(message_type='INTERVENTION',
                          message_content=message_content)

        intervention = Intervention(message=message,
                                    activity_id=activity_id,
                                    created_timestamp=time_intervention)
        return intervention

    def get_intervention_with_real_policy(self, activity_id, current_time, current_day_number):
        """
        Sends an intervention to an agent at a certain time point with a certain message using a learned policy.
        :param activity_id: activity id.
        :param current_time: the current time.
        :param current_day_number: the current day number
        :return: intervention object.
        """
        self.current_time = current_time
        self.current_day_number = current_day_number

        time_intervention = current_time + 1
        message_content = self.define_message_new_intervention(activity_id)

        message = Message(message_type='INTERVENTION',
                          message_content=message_content)

        intervention = Intervention(message=message,
                                    activity_id=activity_id,
                                    created_timestamp=time_intervention)
        return intervention

    def define_message_new_intervention(self, type):
        """
        Returns a message for a new intervention.
        :param type: WORKOUT and SLEEP.
        """
        if type == "workout":
            return "Your activity level has been " \
                   "outstanding the last few days." \
                   "Keep up the good work and don't " \
                   "forget to go for a run today."
        elif type == "sleep":
            return "Your sleep pattern is showing " \
                   "great improvement." \
                   "Keep up the good work and don't " \
                   "forget to go to bed on time tonight."

    def define_time_new_intervention(self, type, seed=False):
        """
        Returns a timestamp for sending the intervention.
        :param type: WORKOUT and SLEEP.
        :param seed: random seed..
        :return: time of new intervention.
        """
        if seed == False:
            if type == "workout":
                return random.choice(self.HOUR_TIME_WORKOUT) * self.NUMBER_SECONDS_HOUR \
                       + self.current_day_number * self.NUMBER_SECONDS_DAY

            elif type == "sleep":
                return random.choice(self.HOUR_TIME_SLEEP) * self.NUMBER_SECONDS_HOUR \
                       + self.current_day_number * self.NUMBER_SECONDS_DAY
        else:
            if type == "workout":
                random.seed(random.random())
                return random.choice(self.HOUR_TIME_WORKOUT) * self.NUMBER_SECONDS_HOUR \
                       + self.current_day_number * self.NUMBER_SECONDS_DAY

            elif type == "sleep":
                random.seed(random.random())
                return random.choice(self.HOUR_TIME_SLEEP) * self.NUMBER_SECONDS_HOUR \
                       + self.current_day_number * self.NUMBER_SECONDS_DAY
