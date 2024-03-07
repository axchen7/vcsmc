from typing import Any, Callable, TypeVar

import tensorflow as tf

T = TypeVar("T", bound=Callable)


def tf_function(*args, **kwargs) -> Callable[[T], T]:
    """
    Like @tf.function, but without masking the original function's type signature.
    """

    return tf.function(*args, **kwargs)  # type: ignore


Tensor = Any
Dataset = Any
