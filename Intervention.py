#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ali el Hassouni
"""

from Message import *


class Intervention:

    def __init__(self, message, activity_id, created_timestamp):
        if message is not None:
            self.activity_id = activity_id
            self.created_at = created_timestamp
            self.message = Message(message_type="INTERVENTION", message_content=message)