from typing import Any, Callable, TypeVar

import tensorflow as tf

_T = TypeVar("_T", bound=Callable)


def tf_function(f: _T) -> _T:
    return tf.function(f)  # type: ignore


Tensor = Any
