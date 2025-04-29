#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run TGAT network intrusion detection system
"""

import os
import sys
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_optimized_main import main

if __name__ == "__main__":
    main()
