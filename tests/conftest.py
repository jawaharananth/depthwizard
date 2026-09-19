"""Shared pytest fixtures. Adds the project root to sys.path so tests can
import project modules (region_footprints, validation, etc.) the same way
every script in this repo already does -- flat imports, no package prefix.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
