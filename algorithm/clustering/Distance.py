#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

import numpy as np
import math
import sklearn
import pandas as pd
import sys


class LB_Keogh:

    def __init__(self):
        self.n_features = 11

    def calculate_distance(self, s1, s2, profile1, profile2, matching):
        """
        Calculate the distance between sequences s1 and s2 using dynamic time warping.
        :param s1: Sequence 1.
        :param s2: Sequence 2.
        :param profile1: Profile 1.
        :param profile2: Profile 2.
        :param matching: @todo
        :return: distance between s1 and s2 based on the dynamic time warping distance metric.
        """
        #return self.distance_week_by_week_dtw(s1, s2, profile1, profile2, number_weeks, matching)
        return self.distance_day_by_day_dtw(s1, s2, profile1, profile2, matching)

    def distance_week_by_week_dtw(self, s1, s2, profile1, profile2, number_weeks, matching):
        """
        Calculate the distance between sequences s1 and s2 using dynamic time warping week by week.
        :param s1: Sequence 1.
        :param s2: Sequence 2.
        :param profile1: Profile 1.
        :param profile2: Profile 2.
        :param number_weeks: The number of weeks to be considered.
        :param matching: @todo
        :return: distance between s1 and s2 based on the dynamic time warping distance metric.
        """
        distance = 0

        for i in range(0, number_weeks):
            s1_week = s1[(i * 11 * 24 * 7): (i + 1) * 11 * 24 * 7]
            s2_week = s2[(i * 11 * 24 * 7): (i + 1) * 11 * 24 * 7]
            distance = distance + self.distance_day_by_day_dtw(s1_week, s2_week, profile1, profile2, matching)
        return distance

    def distance_day_by_day_dtw(self, s1, s2, profile1, profile2, matching):
        """
        Calculate the distance between sequences s1 and s2 using dynamic time warping day by day.
        :param s1: Sequence 1.
        :param s2: Sequence 2.
        :param profile1: Profile 1.
        :param profile2: Profile 2.
        :param matching: @todo
        :return: distance between s1 and s2 based on the dynamic time warping distance metric.
        """
        number_days = math.floor(len(s1 ) /( self.n_features * 24))
        distance = 0

        for i in range(0, number_days):
            #a = matching.get(profile1["scheduling_profile"])[i]
            #b = matching.get(profile2["scheduling_profile"])[i]
            s1_day = s1[(i * self.n_features * 24) : (i + 1) * self.n_features * 24]
            s1_day = np.reshape(s1_day, (-1, self.n_features))
            s1_day = pd.DataFrame(s1_day)
            s2_day = s2[(i * self.n_features * 24) : (i + 1) * self.n_features * 24]
            s2_day = np.reshape(s2_day, (-1, self.n_features))
            s2_day = pd.DataFrame(s2_day)
            distance = distance + self.dynamic_time_warping(s1_day, s2_day)
        return distance

    def distance_hour_by_hour_dtw(self, s1, s2):
        """
        Calculate the distance between sequences s1 and s2 using dynamic time warping hour by hour.
        :param s1: Sequence 1.
        :param s2: Sequence 2.
        :return: distance between s1 and s2 based on the dynamic time warping distance metric.
        """
        distance = 0
        for i in range(0, 24):
            s1_hour = s1[(i * self.n_features) : (i + 1) * self.n_features]
            s2_hour = s2[(i * self.n_features) : (i + 1) * self.n_features]
            distance = distance + self.LB_Keogh(s1_hour, s2_hour, r=1)
        return distance

    def LB_Keogh(self, s1, s2, r):
        """
        Calculate the distance between sequences s1 and s2 using dynamic time warping approximation
        in linear complexity lower bound of Keogh.
        :param s1: Sequence 1.
        :param s2: Sequence 2.
        :param r: radius parameter.
        :return:
        """
        lb_sum = 0
        for i, j in enumerate(s1):

            lb = min(s2[(i - r if i - r >= 0 else 0):(i + r)])
            ub = max(s2[(i - r if i - r >= 0 else 0):(i + r)])

            if j > ub:
                lb_sum = lb_sum + (j - ub) ** 2
            elif j < lb:
                lb_sum = lb_sum + (j - lb) ** 2

        lb_sum = np.sqrt(lb_sum)
        return lb_sum

    def dynamic_time_warping(self, dataset1, dataset2):
        """
        Simple implementation of the dtw. Note that we use the euclidean distance here..
        The implementation follows the algorithm explained in the book very closely.
        Taken from https://github.com/mhoogen/ML4QS/tree/master/Python3Code.
        :param dataset1: dataset 1.
        :param dataset2:  dataset 2.
        :return: the cheapest path.
        """
        # Create a distance matrix between all time points.
        extreme_value = sys.float_info.max
        cheapest_path = np.full((len(dataset1.index), len(dataset2.index)), extreme_value)
        cheapest_path[0,0] = 0

        for i in range(1, len(dataset1.index)):
            for j in range(1, len(dataset2.index)):
                data_row1 = dataset1.iloc[i:i+1,:]
                data_row2 = dataset2.iloc[j:j+1,:]
                d = sklearn.metrics.pairwise.euclidean_distances(data_row1, data_row2)
                cheapest_path[i,j] = d + min(cheapest_path[i-1, j], cheapest_path[i, j-1], cheapest_path[i-1, j-1])

        return cheapest_path[len(dataset1.index)-1, len(dataset2.index)-1]
