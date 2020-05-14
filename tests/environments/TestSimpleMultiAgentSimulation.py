#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ali el Hassouni
"""

import unittest
import environments.MultiAgentSimulation as MultiAgentSimulation


class TestSimpleMultiAgentSimulation(unittest.TestCase):
    def setUp(self):
        self.number_iterations = 100 
        self.number_agents = 5 
        self.time_granularity = 1
        self.SimpleMultiAgentSimulation = MultiAgentSimulation.SimpleMultiAgentSimulation(
            self.number_iterations, self.number_agents, self.time_granularity)
        self.SimpleMultiAgentSimulation.run()

    def test_update_current_time_simulation(self):
        steps = 5
        current_time = self.SimpleMultiAgentSimulation.get_current_time_simulation()
        print (self.SimpleMultiAgentSimulation.get_current_time_simulation())
        self.SimpleMultiAgentSimulation.update_current_time_simulation(steps)
        self.assertEqual(self.SimpleMultiAgentSimulation.get_current_time_simulation(), current_time + steps)

    def test_get_current_time_simulation(self):
        self.assertEqual(self.SimpleMultiAgentSimulation.get_current_time_simulation(),
                         self.SimpleMultiAgentSimulation.current_time_simulation)

    def test_activate_agents(self):
        self.assertEqual(len(self.SimpleMultiAgentSimulation.agents), self.number_agents)


if __name__ == '__main__':
    unittest.main()