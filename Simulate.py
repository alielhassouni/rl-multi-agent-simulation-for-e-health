#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Ali el Hassouni, b.a.kamphorst
"""

import os
import sys
import yaml
import json
from environments import *
from agents import *

if not (len(sys.argv) == 4):
    exit(1)

try:
    configuration_id = int(sys.argv[1]) - 1
    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    __experiments__ = os.path.join(__location__, sys.argv[2])
    experiments = yaml.load(open(__experiments__))
except (FileNotFoundError, ImportError, ValueError):
    exit(2)

for experiment in experiments:
    if not (set(('name', 'agent', 'environment', 'learner')) <= experiment.keys()):
        exit(3)

try:
    __agent_profiles__ = os.path.join(__location__, sys.argv[3])
    print(__agent_profiles__)
    agent_profiles = json.load(open(__agent_profiles__))
    print(agent_profiles)
except (FileNotFoundError, ImportError, ValueError):
    print('AGENT PROFILES NOT FOUND')
    exit(4)


for profile in agent_profiles:
    if not (set(('activity_profiles', 'planning_profiles')) <= agent_profiles.keys()):
        exit(5)

agent_profiles_params = agent_profiles
print(configuration_id)
agent = experiments[configuration_id]['agent']
environment = experiments[configuration_id]['environment']
learner = experiments[configuration_id]['learner']

module = __import__('environments.%s' % environment['class'], fromlist=[environment['class']])
environment_klass = getattr(module, environment['class'])

module = __import__('agents.%s' % agent['class'], fromlist=[agent['class']])
agent_klass = getattr(module, agent['class'])
agent_params = agent

module = __import__('learners.%s' % learner['class'], fromlist=[learner['class']])
learner_klass = getattr(module, learner['class'])
learner_params = learner

if 'write_to_database' in experiments[configuration_id] and experiments[configuration_id]['write_to_database']:
    try:
        from pymongo import MongoClient
        from pymongo.errors import *
        db_connection = MongoClient('localhost', 27017)
        db_name = experiments[configuration_id]['db_name']
    except (ConnectionFailure, ServerSelectionTimeoutError):
        exit(6)

simulation = environment_klass(agent_klass, agent_params, learner_klass, learner_params, agent_profiles_params,
                               environment['number_days_warm_up'], environment['number_days_learn'],
                               environment['number_days_apply'], environment['profile_type'], environment['number_agents'],
                               environment['time_granularity'], environment['results_path'], db_connection, db_name)
simulation.run()
