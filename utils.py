import tensorflow as tf
import os
import constants
from datetime import datetime
import io
import matplotlib.pyplot as plt
from type_utils import Tensor


def create_results_dir():
    """
    Creates the results directory if it does not exist.

    Returns:
        The path to the results directory.
    """

    cur_results_dir = os.path.join(
        constants.RESULTS_DIR_BASE, datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )

    if not os.path.exists(cur_results_dir):
        os.makedirs(cur_results_dir)

    return cur_results_dir


def cur_plt_to_tf_image() -> Tensor:
    """
    Converts the current matplotlib plot to a PNG image and returns it. The plot
    is closed and inaccessible after this call.
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close()
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue())
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
