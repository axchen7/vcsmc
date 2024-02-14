from typing import Any, Callable, TypeVar

import tensorflow as tf


def tf_function(*args, **kwargs):
    """
    Like @tf.function, but without masking the original function's type signature.
    """

    T = TypeVar("T", bound=Callable)

    def decorator(func: T) -> T:
        return func
        return tf.function(func, *args, **kwargs)  # type: ignore

    return decorator


Tensor = Any
