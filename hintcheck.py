"""Runtime type checks for PEP 484 annotations.

>>> from typing import List
>>> @hintchecked
... def f(xs: List[int]) -> int:
...     return max(xs)

>>> f([1, 2])
2

>>> f(1)
Traceback (most recent call last):
    ...
hintcheck.TypeHintError: xs = 1 of type <class 'int'>, \
expected type typing.List[int]
<BLANKLINE>
Type hint in
  File "<doctest hintcheck[1]>", line 1, in f
    @hintchecked
<BLANKLINE>

>>> f(['zzz', 'qqq'])
Traceback (most recent call last):
    ...
hintcheck.TypeHintError: xs[?] = 'zzz' of type <class 'str'>, \
expected type <class 'int'>
<BLANKLINE>
Type hint in
  File "<doctest hintcheck[1]>", line 1, in f
    @hintchecked
<BLANKLINE>

Instead of using the decorator, one can annotate all top-level funtions
using hintcheck_all_functions(). It should be called after all modules
are imported and all functions are defined, e.g. at the beginning
of main().
"""

import os
import sys
import types
import typing
import inspect
import logging
import textwrap
import linecache
import functools
import collections

logger = logging.getLogger(__name__)


class TypeHintError(Exception):
    def __init__(self, *, ctx, expected_type, actual_value):
        self.ctx = ctx
        self.expected_type = expected_type
        self.actual_value = actual_value

    def __str__(self):
        return (
            f'{self.var_name} = {self.actual_value!r} '
            f'of type {type(self.actual_value)}, '
            f'expected type {self.expected_type}\n\n'
            f'Type hint in\n{self.ctx.hint_location.mimic_traceback()}')

    @property
    def var_name(self):
        return self.ctx.var_name


def hintchecked(f):
    hints = typing.get_type_hints(f)
    if not hints:
        return f

    # TODO: what if function is already decorated?
    if hasattr(f, '__code__'):
        code = f.__code__
        hint_location = Location(
            function_name=code.co_name,
            filename=code.co_filename, lineno=code.co_firstlineno)
    else:
        hint_location = Location(
            function_name=str(f), filename=None, lineno=None)

    sig = inspect.signature(f)

    checkers = {
        name: compile_checker(
            t,
            Context(hint_location=hint_location, var_name=name),
            allow_wrap=True)
        for name, t in hints.items()}

    @functools.wraps(f)
    def checked_f(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        for name, value in bound_args.arguments.items():
            if name in checkers:
                bound_args.arguments[name] = checkers[name].check(value)

        result = f(*bound_args.args, **bound_args.kwargs)
        if 'return' in checkers:
            result = checkers['return'].check(result)

        return result

    return checked_f


Location = collections.namedtuple('Location', 'function_name filename lineno')
class Location(Location):
    def mimic_traceback(self):
        if self.filename is None:
            return f'  Function {self.function_name}'

        d = ''
        # attemt to guess where multiline definition ends
        # by balancing parentheses
        for i in range(5):
            d += linecache.getline(self.filename, self.lineno + i)
            if d.count(')') >= d.count('('):
                break

        return (
            f'  File "{self.filename}", '
            f'line {self.lineno}, in {self.function_name}\n' +
            textwrap.indent(textwrap.dedent(d), '    '))


Context = collections.namedtuple('Context', 'hint_location var_name')
class Context(Context):
    def append(self, s):
        return self._replace(var_name=self.var_name + s)


Checker = collections.namedtuple('Checker', 'check is_wrapping')


@functools.singledispatch
def compile_checker(type, ctx, *, allow_wrap):
    def is_instance_check(value):
        if isinstance(value, type):
            return value
        # TODO: There is also complex, Fraction, maybe more.
        # Perhaps add generic way to register 'virtual subtypes'?
        if type is float and isinstance(value, int):
            return value
        raise TypeHintError(ctx=ctx, expected_type=type, actual_value=value)
    return Checker(check=is_instance_check, is_wrapping=False)


@compile_checker.register(typing.TypingMeta)
def _(type, ctx, *, allow_wrap):
    # Generally these types raise exception saying isinstance is not supported,
    # but who knows. We don't want to silently ignore any type errors.
    raise NotImplementedError(type)


@compile_checker.register(typing._Any)
def _(type, ctx, *, allow_wrap):
    return Checker(
        check=lambda x: x,
        is_wrapping=False)


@compile_checker.register(typing._Union)
def _(type, ctx, *, allow_wrap):
    checkers = []
    for t in type.__args__:
        checkers.append(compile_checker(t, ctx, allow_wrap=allow_wrap))

    def union_check(value):
        for c in checkers:
            try:
                return c.check(value)
            except TypeHintError:
                pass
        raise TypeHintError(ctx=ctx, expected_type=type, actual_value=value)

    return Checker(
        check=union_check,
        is_wrapping=any(c.is_wrapping for c in checkers))


@compile_checker.register(typing.TupleMeta)
def _(type, ctx, *, allow_wrap):
    if len(type.__args__) == 2 and type.__args__[1] is ...:
        elem_type, _ = type.__args__
        elem_checker = compile_checker(
            elem_type, ctx.append('[?]'), allow_wrap=False)

        def tuple_ellipsis_check(value):
            if not isinstance(value, tuple):
                raise TypeHintError(
                    ctx=ctx, expected_type=type, actual_value=value)
            for x in value:
                elem_checker.check(x)
            return value

        return Checker(check=tuple_ellipsis_check, is_wrapping=False)

    checkers = []
    for i, t in enumerate(type.__args__):
        checkers.append(
            compile_checker(t, ctx.append(f'[{i}]'), allow_wrap=False))

    def tuple_check(value):
        if not isinstance(value, tuple):
            raise TypeHintError(
                ctx=ctx, expected_type=type, actual_value=value)
        if len(value) != len(checkers):
            raise TypeHintError(
                ctx=ctx, expected_type=type, actual_value=value)
        for c, e in zip(checkers, value):
            c.check(e)
        return value

    return Checker(check=tuple_check, is_wrapping=False)


@compile_checker.register(typing.GenericMeta)
def _(type, ctx, *, allow_wrap):
    if type.__origin__ == typing.List:
        t, = type.__args__

        elem_checker = compile_checker(t, ctx.append('[?]'), allow_wrap=False)

        def list_check(value):
            if not isinstance(value, list):
                raise TypeHintError(
                    ctx=ctx, expected_type=type, actual_value=value)
            for x in value:
                elem_checker.check(x)
            return value

        return Checker(check=list_check, is_wrapping=False)

    elif type.__origin__ == typing.Set:
        # TODO: deduplicate with List.
        t, = type.__args__

        elem_checker = compile_checker(
            t, ctx.append('.some_elem'), allow_wrap=False)

        def set_check(value):
            if not isinstance(value, set):
                raise TypeHintError(
                    ctx=ctx, expected_type=type, actual_value=value)
            for x in value:
                elem_checker.check(x)
            return value

        return Checker(check=set_check, is_wrapping=False)

    elif type.__origin__ == typing.Dict:
        kt, vt = type.__args__

        key_checker = compile_checker(
            kt, ctx.append('.some_key'), allow_wrap=False)
        value_checker = compile_checker(
            vt, ctx.append('.some_value'), allow_wrap=False)

        def dict_check(value):
            if not isinstance(value, dict):
                raise TypeHintError(
                    ctx=ctx, expected_type=type, actual_value=value)
            for k, v in value.items():
                key_checker.check(k)
                value_checker.check(v)
            return value

        return Checker(check=dict_check, is_wrapping=False)

    elif type.__origin__ == typing.Iterator:
        assert allow_wrap  # TODO
        t, = type.__args__
        elem_checker = compile_checker(
            t, ctx.append('.some_elem'), allow_wrap=True)

        def iterator_check(value):
            if not isinstance(value, collections.abc.Iterator):
                raise TypeHintError(
                    ctx=ctx, expected_type=type, actual_value=value)
            return IteratorCheckWrapper(elem_checker, value)

        return Checker(check=iterator_check, is_wrapping=True)

    elif type.__origin__ == typing.Iterable:
        assert allow_wrap  # TODO
        t, = type.__args__
        elem_checker = compile_checker(
            t, ctx.append('.some_elem'), allow_wrap=True)

        def iterable_check(value):
            if not isinstance(value, collections.abc.Iterable):
                raise TypeHintError(
                    ctx=ctx, expected_type=type, actual_value=value)
            return IterableCheckWrapper(elem_checker, value)

        return Checker(check=iterable_check, is_wrapping=True)

    else:
        # Some genericized ABCs have too weak (non-generic) isinstance checks,
        # we don't want to silently ignore type errors because of that.
        raise NotImplementedError(type)


class IteratorCheckWrapper(object):
    def __init__(self, elem_checker, it):
        self.elem_checker = elem_checker
        self.it = it

    def __iter__(self):
        return self

    def __next__(self):
        result = self.it.__next__()
        return self.elem_checker.check(result)


class IterableCheckWrapper(object):
    def __init__(self, elem_checker, iterable):
        self.elem_checker = elem_checker
        self.iterable = iterable

    def __iter__(self):
        return IteratorCheckWrapper(
            self.elem_checker, self.iterable.__iter__())


@compile_checker.register(typing.CallableMeta)
def _(type, ctx, *, allow_wrap):
    assert allow_wrap  # TODO

    arg_types = type.__args__[:-1]
    result_type = type.__args__[-1]

    args_ctx = ctx.append('.args')
    arg_types_tuple = typing.Tuple[arg_types]

    arg_checkers = []
    for i, at in enumerate(arg_types):
        arg_checkers.append(compile_checker(
            at, args_ctx.append(f'[{i}]'), allow_wrap=True))

    def callable_args_check(value):
        if len(value) != len(arg_checkers):
            raise TypeHintError(
                ctx=args_ctx,
                expected_type=arg_types_tuple,
                actual_value=value)
        wrapped_args = []
        for a, c in zip(value, arg_checkers):
            wrapped_args.append(c.check(a))
        return wrapped_args
    args_checker = Checker(check=callable_args_check, is_wrapping=True)

    result_checker = compile_checker(
        result_type, ctx.append('.return'), allow_wrap=True)

    def callable_check(value):
        if not isinstance(value, collections.abc.Callable):
            raise TypeHintError(
                ctx=ctx, expected_type=type, actual_value=value)
        return CallableCheckWrapper(args_checker, result_checker, value)

    return Checker(check=callable_check, is_wrapping=True)


class CallableCheckWrapper(object):
    def __init__(self, args_checker, result_checker, f):
        self.args_checker = args_checker
        self.result_checker = result_checker
        self.f = f

    def __call__(self, *args):
        result = self.f(*self.args_checker.check(args))
        return self.result_checker.check(result)


class DecorableFunction(typing.NamedTuple):
    qualname: str  # including module name
    function: types.FunctionType
    set: typing.Callable[[types.FunctionType], None]


def locate_functions_in_class_or_module(cl_or_m, prefix, visited):
    assert inspect.isclass(cl_or_m) or inspect.ismodule(cl_or_m), cl_or_m
    if cl_or_m in visited:
        return
    else:
        visited.add(cl_or_m)

    for k, v in cl_or_m.__dict__.items():
        if inspect.isfunction(v):
            yield DecorableFunction(
                qualname=prefix + k,
                function=v,
                set=functools.partial(setattr, cl_or_m, k))
        elif inspect.isclass(v):
            yield from locate_functions_in_class_or_module(
                v, prefix + k + '.', visited)

        if inspect.isclass(cl_or_m) and isinstance(v, property):
            for attr in ('fget', 'fset'):
                if hasattr(v, attr):
                    f = getattr(v, attr)
                    if inspect.isfunction(f):
                        def set(new_f):
                            prop = getattr(cl_or_m, k)
                            kwargs = dict(
                                fget=prop.fget, fset=prop.fset,
                                fdel=prop.fdel, doc=prop.__doc__)
                            kwargs[attr] = new_f
                            setattr(cl_or_m, k, property(**kwargs))
                        yield DecorableFunction(
                            qualname=prefix + k + '.' + attr,
                            function=f,
                            set=set)


def locate_all_functions():
    visited = set()
    modules = list(sys.modules.items())
    # Otherwise we get
    # RuntimeError: dictionary changed size during iteration
    for k, v in modules:
        yield from locate_functions_in_class_or_module(v, k + '.', visited)


def locate_all_functions_that_need_hintcheck():
    for f in locate_all_functions():
        if f.qualname.startswith('typing.'):
            continue
        hints = typing.get_type_hints(f.function)
        if hints:
            yield f


def hintcheck_all_functions():
    """Go through all modules and decorate all functions with @hintchecked.

    This is a hack.
    """
    # It's also duplicated in pytest_collection_modifyitems().
    for f in locate_all_functions_that_need_hintcheck():
        logger.info('decorating function %s', f.qualname)
        f.set(hintchecked(f.function))
