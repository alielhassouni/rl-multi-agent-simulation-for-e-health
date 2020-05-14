#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

import numpy as np
from algorithm.clustering.Distance import LB_Keogh
from pprint import pprint
from copy import deepcopy


class KMeans:

    def __init__(self, k=3, tolerance=0.0001, max_iterations=100, verbose=False):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}
        self.classes = {}
        self.LB_Keogh = LB_Keogh()

    def init_centroids(self, data, profiles):
        """
        Initialize centroids by assigning a cluster to each agent.
        :param data: the dataset.
        :param profiles: the agent profiles.
        """
        for i in range(0, self.k):
            self.centroids[i] = {"id": i, "data": data[i], "profile": profiles[i]}

    def init_classes(self):
        """
        Initialize an empty class list.
        """
        for i in range(0, self.k):
            self.classes[i] = []

    def kmeans(self, data, profiles, all_profiles, dtw_days_matching_of_profiles):
        """
        Run K-Means clustering.
        :param data: The data.
        :param profiles: The profiles of the agents.
        :param all_profiles: All profiles possible.
        :param dtw_days_matching_of_profiles: dtw distance matrix.
        """
        self.init_centroids(data, profiles)

        for i in range(0, self.max_iterations):
            print("Number of iteration: " + str(i) + " of " + str(self.max_iterations))
            self.init_classes()
            print(i)

            for s in range(0, len(data)):
                s1 = data[s]
                p1 = profiles[s]
                matching = dtw_days_matching_of_profiles
                distances = [
                    self.LB_Keogh.calculate_distance(
                        s1,
                        self.centroids[centroid]["data"],
                        p1,
                        self.centroids[centroid]["profile"],
                        matching)
                    for centroid in self.centroids]

                classification = distances.index(min(distances))
                self.classes[classification].append(s1)

            previous_centroids = deepcopy(self.centroids)
            pprint(self.classes)

            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous_centroids[centroid]
                current = self.centroids[centroid]
                v = np.sum((current - original_centroid) / original_centroid * 100.0)

                if v > self.tolerance:
                    isOptimal = False

            if isOptimal:
                break

    def predict_all(self, data, profiles, matching):
        """
        Find cluster for new sequence.
        :param data: input data.
        :param profiles: The profiles of the agents.
        :param matching: @todo
        :return: the predictions (i.e. class for each agent).
        """
        for i in range(0, len(data)):
            d = data[i]
            predictions = [LB_Keogh.calculate_distance(d, self.centroids[centroid]["data"], profiles[i],
                                                     self.centroids[centroid]["profile"], matching)
                         for centroid in self.centroids]

        return predictions

    def predict(self, data, p1, matching):
        """
        Find cluster for new sequence.
        :param data: input data.
        :param p1: profile 1.
        :param matching: @todo
        :return: the predictions (i.e. class for each agent).
        """
        distances = [LB_Keogh.calculate_distance(data, self.centroids[centroid]["data"], p1,
                                                 self.centroids[centroid]["profile"], matching)
                     for centroid in self.centroids]

        classification = distances.index(min(distances))
        return classification
