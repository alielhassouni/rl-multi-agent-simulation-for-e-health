#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ali el Hassouni
"""

import unittest
from agents.Agent import *


class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent(1, 'IDLE', 'A') 

    def test_choose_state(self):
        state = self.agent.choose_state()
        self.assertIn(state,['ACTIVE','IDLE'])

    def test_set_current_activity(self):
        self.agent.current_activity = 'A'
        self.assertEqual(self.agent.current_activity,'A')

    def test_choose_activity(self):
        activity = self.agent.choose_activity()
        self.assertIn(activity, ['A', 'B', 'C'])

    def test_set_current_activity_end(self):
        time = self.agent.current_time_simulation
        self.agent.set_current_activity_end(time, 20)
        self.assertEqual(self.agent.current_activity_end, time + 20)

    def test_sample_duration_current_state(self):
        self.assertIn(self.agent.sample_duration_current_state(), range(10,100))

    def test_log_activity(self):
        self.agent.current_state = 'ACTIVE'
        self.assertEqual(self.agent.current_state, 'ACTIVE')
        self.test_log_activity_active()
        self.setUp()
        self.agent.current_state = 'IDLE'
        self.assertEqual(self.agent.current_state, 'IDLE')
        self.test_log_activity_idle()

    def test_log_activity_idle(self):
        self.agent.log_activity_idle()
        self.assertEqual(len(self.agent.activity_log['A']), 1)
        self.assertEqual(len(self.agent.activity_log['B']), 1)
        self.assertEqual(len(self.agent.activity_log['C']), 1)
        self.assertEqual(self.agent.activity_log['A'], [0])
        self.assertEqual(self.agent.activity_log['B'], [0])
        self.assertEqual(self.agent.activity_log['B'], [0])

    def test_log_activity_active(self):
        self.agent.log_activity_active()
        self.assertEqual(len(self.agent.activity_log['A']), 1)
        self.assertEqual(len(self.agent.activity_log['B']), 1)
        self.assertEqual(len(self.agent.activity_log['C']), 1)
        self.assertEqual(self.agent.activity_log['A'], [0.5])
        self.assertEqual(self.agent.activity_log['B'], [0.1])
        self.assertEqual(self.agent.activity_log['C'], [0.7])


if __name__ == '__main__':
    unittest.main()    