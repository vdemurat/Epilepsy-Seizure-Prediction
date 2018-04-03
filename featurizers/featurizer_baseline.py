import numpy as np
import pandas as pd
import scipy.io
import os


def featurization_baseline(data_point):
    nb_features = 8
    features = np.zeros(nb_features)   # nombre de features
    signals_mean = data_point.mean(axis = 0)
        
    features[0] = signals_mean.min()
    features[1] = signals_mean.max()
    features[2] = signals_mean.mean()
    features[3] = signals_mean.std()
    features[4] = signals_mean.var()
    features[5] = np.percentile(signals_mean,25)
    features[6] = np.percentile(signals_mean,50)
    features[7] = np.percentile(signals_mean,75)

    return features