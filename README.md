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

For Python 3.7:
```
git clone https://github.com/Vlad-Shcherbina/hintcheck.git
pip install -e ./hintcheck
```

For Python 3.6:
```
git clone https://github.com/Vlad-Shcherbina/hintcheck.git
cd hintcheck
git checkout py36
pip install -e .
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

## Comparison to mypy

[Mypy](http://mypy-lang.org/) is a static type checker. `hintcheck` checks type annotations at runtime.

Mypy is a large project with some serious support. `hintcheck` is a hobbie thing.

As I understand it, mypy's focus is on finding type errors in large programs before they are run. `hintcheck` focuses on finding lies in type annotations in small programs. Type errors in general are not a concern because Python is mostly good at diagnosing them at runtime, assuming the coverage is good (and there are other reasons to have good coverage anyway).

Mypy supports type annotations in all contexts described in PEP 484 (functions, fields, variables). `hintcheck` only checks annotations in functions and named tuples.

Mypy has no effect on the program execution whatsoever. `hintcheck` adds some overhead, and it's instrumentation has some inevitable subtle side effects. Oh, and of course it raises an exception when the annotation does not match the actual type of the value.

Roughly speaking, mypy requires a contradiction between two type annotations to report an error. If some parts of your codebase are not annotated, annotations on the boundary with these parts are not fully checked. Type inference fills in some of the gaps, but not all. For example, mypy 0.521 in the default mode does not report any problems with the following snippet:
```python
def f(x: int):
    pass

def g(x):
    f(x)

g('z')
```
Mypy also has the strict mode (`--disallow-untyped-calls` and the likes), but with it the main selling point of gradual typing is lost.

`hintcheck`, on the other hand, will not require you to write type annotations just to placate the type checker. You are free to only add annotations where they improve program clarity. In other words, ["use type-hints responsibly"](https://mail.python.org/pipermail/python-dev/2015-May/140104.html). If the annotated function is executed with the value of the wrong type, `hintcheck` will report an error even if the rest of the program in not annotated at all.

