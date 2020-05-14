#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

import numpy as np


class Activity:

    def __init__(self, label, duration_mean, duration_sd, start_time_activity=None,
                 end_time_activity=None, intervention=None, fatigue=None):
        self.label = label
        self.duration_mean = duration_mean
        self.duration_sd = duration_sd
        self.fatigue = fatigue
        self.duration = self.sample_activity_duration(self.duration_mean, self.duration_sd,
                                                      self.fatigue)
        self.start_time_activity = start_time_activity
        self.end_time_activity = self.start_time_activity + self.duration
        self.intervention = intervention

    def sample_activity_duration(self, mean, sd, fatigue):
        """
        Sample a duration for an activity from a normal distribution
        with duration_mean as the mean and duration_sd as the standard deviation.
        :param mean: Mean of duration of activity.
        :param sd: Standard deviation of duration of activity.
        :param fatigue: Fatigue level.
        :return: Rounded sampled duration.
        """
        if fatigue == None or fatigue == 0:
            self.duration = np.random.normal(mean, sd)
        else:
            self.duration = (1/np.sqrt(int(fatigue)))*np.random.normal(mean, sd)
        return round(self.duration)

    def update_end_time_activity(self, start_time_activity):
        """
        Update end time of activity.
        :param start_time_activity: Start time of activity
        """
        self.end_time_activity = start_time_activity + self.duration
