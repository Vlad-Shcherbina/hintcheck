import textwrap
import linecache

import pytest

import hintcheck


def pytest_collection_modifyitems(items):
    hintcheck.hintcheck_all_functions()
