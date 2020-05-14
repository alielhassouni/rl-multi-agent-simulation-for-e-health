#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni
"""

from pymongo import MongoClient
import matplotlib
matplotlib.use('agg')
import pandas
from datetime import datetime

db_connection = MongoClient('localhost', 27017)
query = {'activities_log': {'$exists': 1},
         'scheduling_profile': {'$exists': 1},}
projection = {'_id': 0, 'activities_log': 1, 'scheduling_profile': 1}
sort = [('agent_id', 1)]
db_names = {"LSPI": "experiment_101_LSPI_Normal_Grouped_KMedoids_DTW"}
col_names = ['Activity', 'Hour']
result = pandas.DataFrame(columns=col_names)

for key, val in db_names.items():
    db_name = val
    db = db_connection[db_name]
    query_result = db.agent_state.find(query, projection)
    counter = 0

    for i in range(0, query_result.count()):
            profile = query_result[i]['scheduling_profile']
            nhours = len(query_result[i]["activities_log"])

            for j in range(0, nhours):
                activity = query_result[i]["activities_log"][j][0]
                timestamp = datetime.strptime(query_result[i]["activities_log"][j][1], '%Y-%m-%d %H:%M:%S')
                hour = timestamp.hour + (timestamp.minute/60)

                if profile == "type_1_tight_schedule":
                    activity_profile = activity + "_Workaholic"
                elif profile == "type_2_normal_schedule":
                    activity_profile = activity + "_Arnold"
                elif profile == "type_3_light_schedule":
                    activity_profile = activity + "_Retiree"

                result.loc[counter] = [activity_profile, hour]
                counter += 1
print(result)
result.to_csv('joy_plot_data.csv', sep=';', encoding='utf-8')
