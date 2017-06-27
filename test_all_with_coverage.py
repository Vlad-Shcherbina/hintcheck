if __name__ == '__main__':
    import os
    import sys
    import webbrowser

    import pytest
    # requires pytest-cov
    res = pytest.main(['--cov', '--cov-report=html'] + sys.argv[1:])
    webbrowser.open('file://' + os.path.abspath('htmlcov/index.html'))
