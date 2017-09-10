# hintcheck

[PEP 484](https://www.python.org/dev/peps/pep-0484/) standardizes the meaning of type annotations.
However, type annotations are ignored by the interpreter.
When type annotations are used as a form of documentation, it would be nice to know that they are in sync with the code.

`hintcheck` is a library that checks type annotations against the actual types of the values at runtime.

```python
def f(x: int):
    return 3 * x
```
```
>>> f('z')  # without hintcheck
'zzz'
```
```pytb
>>> f('z')  # with hintcheck
Traceback (most recent call last):
...
hintcheck.TypeHintError: x = 'z' of type <class 'str'>, expected type <class 'int'>

Type hint in
  File "example.py", line 1, in f
    def f(x: int):
```
## Installation

```
git clone https://github.com/Vlad-Shcherbina/hintcheck.git
pip install -e ./hintcheck
```


## Usage

### With pytest

`hintcheck` is enabled by default as a `pytest` plugin.
To run tests without `hintcheck`, use the following command:
```
pytest -p no:pytest-hintcheck
```

### In executable scripts

```python
import hintcheck
hintcheck.monkey_patch_named_tuple_constructors()
# This should be called before any other modules are imported.

...

if __name__ == '__main__':
    hintcheck.hintcheck_all_functions()
    # This should be called after all modules are imported,
    # but before the program is run.

    main()

```

## Limitations

Lots of them.
