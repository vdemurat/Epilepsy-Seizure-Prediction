#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 14:30:50 2018

@author: saad
"""
import numpy as np


def featurize(X,cutoff=False,T=15):
    X=np.transpose(X)
    N=len(X)
    sp=int(10*60/T)
    print(type(sp))
    print(X.shape)
    split_array=np.floor(np.linspace(0,N,sp+1))
    split_array=split_array[1:sp]
    split_array=split_array.astype('int')
    separate=np.split(X,split_array)
    e=[]
    print(separate[0].shape)
    for tranche in separate: 
        energies=[]
        tranche=np.transpose(tranche)
        for i in range(16): 
            #print(tranche[i].shape)
            print(tranche.shape)
            FT=np.fft.fft(tranche[i])
            energy=np.log(np.linalg.norm(FT)**2)
            energies.append(energy)
        e.append(np.array(energies))
    FF=np.fft.fft(X)    
    total_energy=np.log(compute_norm(FF))
    mean_energy=np.mean(e,axis=0)
    std_energy=np.std(e,axis=0)
    covariance=np.cov(np.transpose(e))
    covariance=np.array(covariance)
    e=np.array(e)
    ##reshaping stuff
    e=e.reshape(-1)
    covariance=covariance.reshape(-1)
    return np.hstack((total_energy,np.array(e),mean_energy,std_energy,covariance))


def featurize_weak(X,cutoff=False,T=15): 
    X=np.transpose(X)
    N=len(X)
    sp=int(10*60/T)
    print(type(sp))
    print(X.shape)
    split_array=np.floor(np.linspace(0,N,sp+1))
    split_array=split_array[1:sp]
    split_array=split_array.astype('int')
    separate=np.split(X,split_array)
    e=[]
    for tranche in separate: 
        energies=[]
        #print(tranche.shape)
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
    return np.hstack((total_energy,np.array(e),mean_energy))    


def featurize_FF(X,cutoff=50): 
    transforms=[]
    for column in X: 
        FF=np.fft.fft(X)
        FFshort=FF[0:cutoff]
        transforms.append(FFshort)



def compute_norm(signal):
    signal=np.transpose(signal)
    norms=[]
    for x in signal:
        norms.append(np.linalg.norm(x)**2)
    return np.mean(norms)



        