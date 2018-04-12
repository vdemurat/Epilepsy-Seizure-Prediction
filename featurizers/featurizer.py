#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:30:50 2018

@author: saad
"""
import numpy as np


def featurize(X,cutoff=False,T=15):
    N=len(X)
    sp=10*60/T
    separate=np.split(X,sp)
    e=[]
    for tranche in separate: 
        energies=[]
        for i in range(16): 
            FT=np.fft.fft(tranche[i])
            energy=np.linalg.norm(FT)**2
            energies.append(energy)
        e.append(np.array(energies))
    FF=np.fft.fft(X)    
    total_energy=compute_norm(FF)
    mean_energy=np.mean(e,axis=0)
    std_energy=np.std(e,axis=0)
    covariance=np.cov(np.transpose(e))
    covariance=np.array(covariance)
    e=np.array(e)
    ##reshaping stuff
    e=e.reshape(-1)
    lc=covariance.shape[0]*covariance.shape[1]
    covariance=covariance.reshape(-1)
    return np.hstack((total_energy,np.array(e),mean_energy,std_energy,covariance))


def featurize_weak(X,cutoff=False,T=15): 
    N=len(X)
    sp=10*60/T
    separate=np.split(X,sp)
    e=[]
    for tranche in separate: 
        energies=[]
        for i in range(16): 
            FT=np.fft.fft(tranche[i])
            energy=np.linalg.norm(FT)**2
            energies.append(energy)
        e.append(np.array(energies))
    e=np.array(e)
    FF=np.fft.fft(X)    
    total_energy=compute_norm(FF)
    mean_energy=np.mean(e,axis=0)
    e=e.reshape(-1)
    return np.hstack(total_energy,np.array(e),mean_energy)    



        