# -*- Python -*-

__all__ = []

"""
decorator to inherit docstring and keep the signature
code adapted from : http://code.activestate.com/recipes/576862/

Usage:

class Foo(object):
    # the name must be with an underscore.
    def _foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @DocInherit
    @keepingArgs
    def foo(self, a, b):
        self._foo()

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""

from functools import wraps
import decorator


class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = "_" + mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=("__name__", "__module__"))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=("__name__", "__module__"))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func


def keepingArgs(target):
    # the target function has been prepended to the list of arguments
    def wrapper(target, *args, **kwargs):
        return target(*args, **kwargs)

    # We are calling the returned value with the target function to get a
    # 'proper' wrapper function back
    return decorator.decorator(wrapper)(target)
