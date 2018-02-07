import pytest

pytest_plugins = "pytester"  # to get testdir fixture


def is_plugin_installed():
    import pkg_resources
    for ep in pkg_resources.iter_entry_points('pytest11'):
        if ep.name == 'pytest-hintcheck':
            return True
    return False


@pytest.mark.skipif(
    not is_plugin_installed(),
    reason='requires hintcheck plugin to be installed')
def test_plugin(testdir):
    testdir.makepyfile("""
    import typing

    def f(x: str):
        pass

    def test_function_example():
        f(42)

    class Pt(typing.NamedTuple):
        x: int
        y: int

    def test_tuple_example():
        Pt(2, 'qqq')
    """)

    result = testdir.runpytest_subprocess('--verbose', '--color=no')
    result.stdout.fnmatch_lines(
        "* hintcheck.TypeHintError: x = 42 of type <class 'int'>, "
        "expected type <class 'str'>*")
    result.stdout.fnmatch_lines(
        "* hintcheck.TypeHintError: y = 'qqq' of type <class 'str'>, "
        "expected type <class 'int'>*")
