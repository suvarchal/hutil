#!/usr/bin/env python
# coding: utf-8
"""
Pytest configuration for hutil tests.
"""

import os
import sys
import pytest

# Add the parent directory to the path so that hutil can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
