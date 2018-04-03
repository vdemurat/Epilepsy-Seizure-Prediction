import numpy as np
import pandas as pd
import os
import random



def oversample(preic_data, size_inte, method = 'copy', k_nearest=5):
    ratio = int(size_inte/len(preic_data))
    
    if method == 'copy':
        new_data = preic_data.copy()
        for i in range(ratio-2):
            new_data = new_data + preic_data
        
    elif method == 'SMOTE':
        new_data = SMOTE(preic_data, ratio, k_nearest)
        
    np.random.shuffle(new_data)     
    return new_data


def SMOTE(preic_data, ratio, k_nearest):
    nb_samples_per_neighbor = int((ratio-2)/k_nearest)
    new_data = preic_data.copy()
    
    for i, current in enumerate(preic_data):
        dist = np.zeros(len(preic_data))
        for j in range(len(preic_data)):
            dist[j] = np.linalg.norm(preic_data[i]-preic_data[j], 2)    #many distance metrics possibles
            
        closest = np.argsort(dist)
        
        for neighbor in closest[1:(k_nearest+1)]:
            r_factors = np.random.uniform(0.0, 1.0, size=nb_samples_per_neighbor)
            for k in range(nb_samples_per_neighbor):
                new_point = current + r_factors[k]*(preic_data[neighbor]-current)
                new_data = new_data + [new_point]
            
    return new_data