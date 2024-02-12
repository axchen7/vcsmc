import tensorflow as tf
from typing import Any, Callable, TypeVar

_T = TypeVar("_T", bound=Callable)


def tf_function(f: _T) -> _T:
    return tf.function(f)  # type: ignore


Tensor = Any
