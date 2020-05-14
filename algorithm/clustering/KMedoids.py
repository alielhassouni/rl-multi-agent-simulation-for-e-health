#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
Inspired by and based on https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py
"""

import numpy as np
from algorithm.clustering.Distance import LB_Keogh
from pprint import pprint
from copy import deepcopy
import random


class KMedoids:

    def cluster(self, distances, k=3):
        """
        Peroform K-Medoid clustering
        :param distances: distance matrix.
        :param k: parameter k (i.e. number of clusters).
        :return: clusterings.
        """
        m = distances.shape[0]

        # Pick k random medoids.
        curr_medoids = np.array([-1] * k)

        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])

        old_medoids = np.array([-1] * k)
        new_medoids = np.array([-1] * k)

        while not ((old_medoids == curr_medoids).all()):
            clusters = self.assign_points_to_clusters(curr_medoids, distances)

            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = self.compute_new_medoid(cluster, distances)

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]

        return clusters, curr_medoids

    def assign_points_to_clusters(self, medoids, distances):
        """
        Assign points to clusters.
        :param medoids: the medoids found.
        :param distances: the distances.
        :return: cluster assignments.
        """
        distances_to_medoids = distances[:, medoids]
        clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters

    def compute_new_medoid(self, cluster, distances):
        """
        Compute new medoids.
        :param cluster: the current clusters.
        :param distances: the distances.
        :return: cluster distances.
        """
        mask = np.ones(distances.shape)
        mask[np.ix_(cluster, cluster)] = 0.
        cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)

