import numpy as np
import pandas as pd
import scipy.io
import os

def featurization(data_point,offset):
    return offset + data_point.mean()