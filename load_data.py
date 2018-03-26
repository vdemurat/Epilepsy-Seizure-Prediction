import numpy as np
import pandas as pd
import scipy.io
import os

def load_featurize_folder(folder_path,folder,featurize_func,args):
    filenames = os.listdir(folder_path + folder + '/')
    filenames_interictal = []
    filenames_preictal = []
    filenames_test = []

    for i in filenames:
        len_interictal = len(folder) + len('_interictal')
        len_preictal = len(folder) + len('_preictal')
        len_test = len(folder) + len('_test')
        if len_interictal >=  len(folder + '_interictal') and i[:len_interictal] == folder+'_interictal':
            filenames_interictal.append(i)
        if len_preictal >=  len(folder + '_preictal') and i[:len_preictal] == folder+'_preictal':
            filenames_preictal.append(i)
        if len_test >=  len(folder + '_test') and i[:len_test] == folder+'_test':
            filenames_test.append(i)

    filenames_interictal.sort()
    filenames_preictal.sort()
    filenames_test.sort()
    interictal_data = []
    preictal_data = []
    test_data = []

    for i in filenames_interictal:
        seg = scipy.io.loadmat(folder_path+ folder + '/'+ i)
        key = [i for i in list(seg.keys()) if i.startswith('interictal')][0]
        features = featurize_func(seg[key][0][0][0],**args)
        interictal_data.append(features)
    for i in filenames_preictal:
        seg = scipy.io.loadmat(folder_path+ folder + '/' + i)
        key = [i for i in list(seg.keys()) if i.startswith('preictal')][0]
        features = featurize_func(seg[key][0][0][0],**args)
        preictal_data.append(features)
    for i in filenames_test:
        seg = scipy.io.loadmat(folder_path+ folder+ '/' + i)
        key = [i for i in list(seg.keys()) if i.startswith('test')][0]
        features = featurize_func(seg[key][0][0][0],**args)
        test_data.append(features)
    return interictal_data, preictal_data, test_data


def load_featurize_folders(folder_path,list_folders,featurize_func,args):
    interictal_data = []
    preictal_data = []
    test_data = []
    for folder in list_folders:
        print('loading: ',folder)
        int_folder, preic_folder, test_folder = load_featurize_folder(folder_path, folder, featurize_func, args)
        interictal_data = interictal_data + int_folder
        preictal_data = preictal_data + preic_folder
        test_data = test_data + test_folder
    return interictal_data, preictal_data, test_data