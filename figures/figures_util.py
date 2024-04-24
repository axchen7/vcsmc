import os
import sys


def set_path():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir("..")
    sys.path.append(".")


def make_output_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(cur_dir, "output")
    os.mkdir(output_dir)
    return output_dir
