#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

import datetime


class Features:

    def day_of_week(self, current_time):
        """
        Returns day of the week of the timestamp
        :param current_time: 
        :return: a string day of the week 
        """
        return datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S").strftime('%a')