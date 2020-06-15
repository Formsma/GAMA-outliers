#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:18:39 2019

Compute the distance scores of GAMA spectra using unsupervised
random forest.

@author: Job Formsma
"""

import numpy as np
import pandas as pd                   
import sklearn.ensemble               
from astropy.io import fits           

from tqdm import tqdm                 
from glob import glob                 
import time

from multiprocessing import Pool
from functions import synthetic_data, labeller, merge_RF

# Data paths
PATH = "../data/arrays/"        # Data location
NAME = "SPECS_107532_SN2_5"     # File name

# Processing variables
N_PROC = 64                           # Processes (64 for norma4)
BINS = 4                              # Signal to Noise ratio bins
CHUNKS = 7                            # Redundant chunks
SIZE = 10000                          # Size of chunks
N_TRAIN = 100                         # Decision trees per chunk
RUNS = 1                              # Amount of runs of this algorithm

    
def worker_apply(estimator):
    """Utilizes global variables data_cut and iterator i"""
    return estimator.predict(data_all) == 0

    
def worker_similarity(i):
    """Utilizes global variables AM and true"""
    ret = np.zeros(AM.shape[0])
    ret[i:] = np.sum((AM[i] != -1) & (AM[i] == AM[i:]), axis=1) / true[i]
    #return np.sum((AM[i] != -1) & (AM[i] == AM), axis=1) / true[i]
    return ret

# Load spectra names of full data set
specids = np.genfromtxt(PATH+NAME+".txt", unpack=True, dtype='str')

# Load query with information about all spectra
query = pd.read_csv("GAMA_SpecAll.csv")

# Load spectra data
data_all = np.memmap(PATH+NAME+".array", dtype='float16', mode='r', 
                     shape=(len(specids), 8000))
                 
# Cut data into signal to noise ratio bins
data_cut = np.array_split(data_all, BINS)

# Time name
t = "{:.0f}".format(time.time())
print(t)

RF = []                                  # Prepare storage RFs 

# Insert into name of output file
start = 2.5
end   = 110
    
# Work on each data set
for i, data in enumerate(tqdm(data_cut, desc="subset ")):

    data_syn = synthetic_data(data_cut[i])   # Create synthetic data
    d, l = labeller(data_cut[i], data_syn)   # Label the data

    # Create RF for each chunk
    for _ in tqdm(range(CHUNKS), desc="chunks "):

        # Make subset of the data
        choice = np.random.choice(len(l), SIZE, replace=False)
        subset_d = d[choice]
        subset_l = l[choice]

        # Create Random Forest
        RFC = sklearn.ensemble.RandomForestClassifier(n_estimators=N_TRAIN, n_jobs=max(1, N_PROC))#, max_features=500)
        RFC.fit(subset_d, subset_l)
        RF.append(RFC)
        
RF = merge_RF(RF, N_PROC)       # Merge RFs from all chunks
AM = RF.apply(data_all)         # Apply RF on subset
 
# Find wrongly labelled outcomes in the Apply Matrix
with Pool(processes=N_PROC) as pool:
    check = list(tqdm(pool.imap(worker_apply, RF.estimators_), 
                      total=AM.shape[1], desc="apply  ", smoothing=0))

AM[np.transpose(check) == False] = -1          # Mask wrongly labelled points
true = np.sum(AM != -1, axis=1)  # Normalization factor

# Get pair-wise similarity scores
with Pool(processes=N_PROC) as pool:
    sim_mat = list(tqdm(pool.imap(worker_similarity, 
                                  range(len(AM))), 
                        total=len(AM), 
                        desc="sim mat"))

# Copy upper triangle of simmat to lower
sim_mat = np.asarray(sim_mat)
i_lower = np.tril_indices(sim_mat.shape[0], -1)
sim_mat[i_lower] = sim_mat.T[i_lower]

# Copy simmilarity matrix to array map for t-SNE
#sim_mat_map = np.memmap("/scratch/users/formsma/storage/data/matrix/sim_{}.array".format(NAME), dtype="float16", mode="w+", shape=sim_mat.shape)
#sim_mat_map[:] = sim_mat
        
# Compute distance matrix
dis_mat = 1 - sim_mat

scores = np.sum(dis_mat, axis=1) / len(AM)

# Sort spectra names according to scores
weirdest = np.array([s for _, s in sorted(zip(scores, specids), reverse=True)])
scores = np.array(sorted(scores, reverse=True))

# Print the top 10
for w, s in zip(weirdest[:10], scores[:10]):
    print(w)

# Save output to file
np.savetxt("output/"+NAME+"_{:.2f}_{:.2f}_{}_scores.txt".format(start, end, t), np.transpose([weirdest, scores]), fmt="%s %s")
np.savetxt("output/"+NAME+"_{:.2f}_{:.2f}_{}_features.txt".format(start, end, t), np.transpose(RF.feature_importances_))