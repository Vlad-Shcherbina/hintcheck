import textwrap
import linecache

import pytest

import hintcheck


def pytest_configure(config):
    hintcheck.monkey_patch_named_tuple_constructors()


def pytest_collection_modifyitems(items):
    hintcheck.hintcheck_all_functions()
