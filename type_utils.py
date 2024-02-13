from typing import Any, Callable, TypeVar

import tensorflow as tf


def tf_function(*args, **kwargs):
    T = TypeVar("T", bound=Callable)

    def decorator(func: T) -> T:
        return tf.function(func, *args, **kwargs)  # type: ignore

    return decorator


Tensor = Any
