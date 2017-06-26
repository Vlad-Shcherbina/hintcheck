pytest_plugins = "pytester"  # to get testdir fixture


def test_plugin(testdir):
    # requires hintcheck plugin to be installed

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
