#! python3
import sys
import collections
from typing import (
    TypeVar, Any, Union, Tuple, NamedTuple, List, Set, Dict,
    Iterator, Iterable, Callable, Deque, SupportsInt)
import logging
import inspect

import pytest

from hintcheck import (
    hintchecked,
    TypeHintError,
    locate_all_functions_that_need_hintcheck,
    hintcheck_all_functions,
    monkey_patch_named_tuple_constructors)


def test_smoke():
    @hintchecked
    def f(x: int, y=None) -> int:
        return y or 2 * x

    assert f(42) == 84
    assert f(42, y=1) == 1

    with pytest.raises(TypeHintError) as exc_info:
        f('zzz')
    assert exc_info.value.var_name == 'x'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 'zzz'

    with pytest.raises(TypeHintError) as exc_info:
        f(42, y='zzz')
    assert exc_info.value.var_name == 'return'
    assert exc_info.value.expected_type == int


def test_no_annotations():
    def orig_f(x):  # pragma: no cover
        return x
    f = hintchecked(orig_f)
    assert f is orig_f


def test_type_not_supported():
    with pytest.raises(NotImplementedError) as exc_info:
        @hintchecked
        def f(x: 42):
            pass
    assert '42 does not look like type' in str(exc_info.value)

    with pytest.raises(NotImplementedError) as exc_info:
        @hintchecked
        def f(x: Deque[int]):
            pass
    assert str(exc_info.value).startswith('typing.Deque[int]\n\n')


def test_typevar():
    T = TypeVar('T')
    @hintchecked
    def f(x: T):
        pass

    f(42)


def test_unparameterized():
    @hintchecked
    def f(x: Set):
        pass
    f({1, 2, 3})
    with pytest.raises(TypeHintError) as exc_info:
        f(42)
    assert exc_info.value.var_name == 'x'
    assert exc_info.value.expected_type == Set


def test_any():
    @hintchecked
    def f(x: Any):
        return x

    assert f(42) == 42
    assert f('zzz') == 'zzz'


def test_number_subtyping():
    @hintchecked
    def f(i: int, f: float):
        pass

    f(i=1, f=2.0)
    f(i=1, f=2)

    with pytest.raises(TypeHintError) as exc_info:
        f(i=1.0, f=2.0)
    assert exc_info.value.var_name == 'i'
    assert exc_info.value.expected_type == int


def test_union():
    @hintchecked
    def f(x: Union[int, float]):
        return 2 * x

    assert f(42) == 84
    assert f(42.0) == 84.0

    with pytest.raises(TypeHintError) as exc_info:
        f('zzz')
    assert exc_info.value.var_name == 'x'
    assert exc_info.value.expected_type == Union[int, float]


def test_tuple():
    @hintchecked
    def f(t: Tuple[int, str]):
        return t

    assert f(t=(42, 'zzz')) == (42, 'zzz')

    with pytest.raises(TypeHintError) as exc_info:
        f(t=42)
    assert exc_info.value.var_name == 't'
    assert exc_info.value.expected_type == Tuple[int, str]

    with pytest.raises(TypeHintError) as exc_info:
        f(t=(42,))
    assert exc_info.value.var_name == 't'
    assert exc_info.value.expected_type == Tuple[int, str]

    with pytest.raises(TypeHintError) as exc_info:
        f(t=(42, 'zzz', 43))
    assert exc_info.value.var_name == 't'
    assert exc_info.value.expected_type == Tuple[int, str]

    with pytest.raises(TypeHintError) as exc_info:
        f(t=('q', 'zzz'))
    assert exc_info.value.var_name == 't[0]'
    assert exc_info.value.expected_type == int


def test_tuple_ellipsis():
    @hintchecked
    def f(t: Tuple[int, ...]):
        return t

    assert f(t=()) == ()
    assert f(t=(1,)) == (1,)
    assert f(t=(1, 2)) == (1, 2)

    with pytest.raises(TypeHintError) as exc_info:
        f(t=42)
    assert exc_info.value.var_name == 't'
    assert exc_info.value.expected_type == Tuple[int, ...]

    with pytest.raises(TypeHintError) as exc_info:
        f(t=('zzz',))
    assert exc_info.value.var_name == 't[?]'
    assert exc_info.value.expected_type == int


def test_named_tuple():
    monkey_patch_named_tuple_constructors()

    filename = inspect.stack()[0].filename
    t_lineno = inspect.stack()[0].lineno + 2

    class T(NamedTuple):
        x: int
        y: int
        z: int = 0

    ts_lineno = inspect.stack()[0].lineno + 1
    TS = NamedTuple('TS', [('x', str), ('y', str)])

    t = T(42, 43)
    assert t == (42, 43, 0)
    assert t.x == 42
    assert t[1] == 43
    T(*range(2))
    T(x=42, y=42)
    TS('hello', 'world')

    with pytest.raises(TypeHintError) as exc_info:
        T(42, 'zzz')
    assert exc_info.value.var_name == 'y'
    assert exc_info.value.expected_type == int
    assert exc_info.value.ctx.hint_location.function_name == 'T'
    assert exc_info.value.ctx.hint_location.filename == filename
    assert exc_info.value.ctx.hint_location.lineno == t_lineno

    with pytest.raises(TypeHintError) as exc_info:
        TS(42, 'zzz')
    assert exc_info.value.var_name == 'x'
    assert exc_info.value.expected_type == str
    assert exc_info.value.ctx.hint_location.function_name == 'TS'
    assert exc_info.value.ctx.hint_location.filename == filename
    assert exc_info.value.ctx.hint_location.lineno == ts_lineno


def test_list():
    @hintchecked
    def f(xs: List[int]):
        return xs

    assert f([1, 2]) == [1, 2]
    assert f([]) == []

    with pytest.raises(TypeHintError) as exc_info:
        f(42)
    assert exc_info.value.var_name == 'xs'
    assert exc_info.value.expected_type == List[int]

    with pytest.raises(TypeHintError) as exc_info:
        f([1, 'z'])
    assert exc_info.value.var_name == 'xs[?]'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 'z'

    with pytest.raises(TypeHintError) as exc_info:
        f((1, 2))
    assert exc_info.value.var_name == 'xs'
    assert exc_info.value.expected_type == List[int]


def test_set():
    @hintchecked
    def f(s: Set[int]):
        return s

    assert f(set()) == set()
    assert f({1, 2}) == {1, 2}

    with pytest.raises(TypeHintError) as exc_info:
        f(42)
    assert exc_info.value.var_name == 's'
    assert exc_info.value.expected_type == Set[int]

    with pytest.raises(TypeHintError) as exc_info:
        f({'z'})
    assert exc_info.value.var_name == 's.some_elem'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 'z'

    with pytest.raises(TypeHintError) as exc_info:
        f(frozenset([1, 2]))
    assert exc_info.value.var_name == 's'
    assert exc_info.value.expected_type == Set[int]


def test_dict():
    @hintchecked
    def f(d: Dict[int, str]):
        return d

    assert f({42: 'zzz'}) == {42: 'zzz'}
    assert f({}) == {}

    with pytest.raises(TypeHintError) as exc_info:
        f(42)
    assert exc_info.value.var_name == 'd'
    assert exc_info.value.expected_type == Dict[int, str]

    with pytest.raises(TypeHintError) as exc_info:
        f({1: 2})
    assert exc_info.value.var_name == 'd.some_value'
    assert exc_info.value.expected_type == str
    assert exc_info.value.actual_value == 2

    with pytest.raises(TypeHintError) as exc_info:
        f({'z': 'y'})
    assert exc_info.value.var_name == 'd.some_key'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 'z'


def test_iterator():
    @hintchecked
    def f(it: Iterator[int]):
        return it

    t = f(iter([1, 2, 3]))
    assert isinstance(t, collections.abc.Iterator)
    assert list(t) == [1, 2, 3]

    with pytest.raises(TypeHintError) as exc_info:
        f(42)
    assert exc_info.value.var_name == 'it'
    assert exc_info.value.expected_type == Iterator[int]

    t = f(iter('abc'))
    with pytest.raises(TypeHintError) as exc_info:
        list(t)
    assert exc_info.value.var_name == 'it.some_elem'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 'a'


def test_iterable():
    @hintchecked
    def f(it) -> Iterable[int]:
        return it

    t = f([1, 2, 3])
    assert isinstance(t, collections.abc.Iterable)
    assert list(t) == [1, 2, 3]
    assert list(t) == [1, 2, 3]  # reusable

    with pytest.raises(TypeHintError) as exc_info:
        f(42)
    assert exc_info.value.var_name == 'return'
    assert exc_info.value.expected_type == Iterable[int]

    t = f('abc')
    with pytest.raises(TypeHintError) as exc_info:
        list(t)
    assert exc_info.value.var_name == 'return.some_elem'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 'a'


def test_callable():
    @hintchecked
    def f(x, cb: Callable[[int], str]):
        return cb(*x)

    assert f([1], lambda x: x * 'hw') == 'hw'

    with pytest.raises(TypeHintError) as exc_info:
        f([1], 42)
    assert exc_info.value.var_name == 'cb'
    assert exc_info.value.expected_type == Callable[[int], str]
    assert exc_info.value.actual_value == 42

    with pytest.raises(TypeHintError) as exc_info:
        f([1, 2], lambda x, y: 'hw')  # pragma: no cover
    assert exc_info.value.var_name == 'cb.args'
    assert exc_info.value.expected_type == Tuple[int]
    assert exc_info.value.actual_value == (1, 2)

    with pytest.raises(TypeHintError) as exc_info:
        f([0.5], lambda x: 'hw')  # pragma: no cover
    assert exc_info.value.var_name == 'cb.args[0]'
    assert exc_info.value.expected_type == int
    assert exc_info.value.actual_value == 0.5

    with pytest.raises(TypeHintError) as exc_info:
        f([1], lambda x: 42)
    assert exc_info.value.var_name == 'cb.return'
    assert exc_info.value.expected_type == str
    assert exc_info.value.actual_value == 42


def test_callable_no_wrap():
    @hintchecked
    def f(c) -> Tuple[int, Callable[[], None]]:
        return 42, c

    f(lambda: None)
    f(lambda x: x)  # passes because these is no wrapping

    with pytest.raises(TypeHintError) as exc_info:
        f('zzz')
    assert exc_info.value.var_name == 'return[1]'
    assert exc_info.value.expected_type == Callable[[], None]
    assert exc_info.value.actual_value == 'zzz'


def f(x: int):
    return x


class C:
    def m(self, x: int):
        return x

    @property
    def p(self) -> int:
        return 'zzz'

    @property
    def rw(self) -> int:
        return 'zzz'

    @rw.setter
    def rw(x, value: int):  # pragma: no cover
        pass


def test_locate_all_functions_that_need_hintcheck():
    # pytest imports this module twice, first as '__main__'
    # and then as 'hintcheck_test', so unless we take measures
    # we will see two copies of f and C.
    fs = {
        f.function
        for f in locate_all_functions_that_need_hintcheck()
        if f.function.__module__ == 'hintcheck_test'}
    for q in fs:
        print(q, q.__module__)
    assert fs == {f, C.m, C.p.fget, C.rw.fget, C.rw.fset}


def test_hintcheck_all_functions():
    f('zz')
    C().m('zz')

    hintcheck_all_functions()

    with pytest.raises(TypeHintError):
        f('zz')
    with pytest.raises(TypeHintError):
        C().m('zz')
    with pytest.raises(TypeHintError):
        C().p
    with pytest.raises(TypeHintError):
        C().rw
    with pytest.raises(TypeHintError):
        C().rw = 'zz'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    pytest.main([
        __file__,
        __file__.replace('_test.py', '.py'),  # for doctest
        ] + sys.argv[1:])
