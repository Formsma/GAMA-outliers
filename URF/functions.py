import numpy as np
from astropy.io import fits


def synthetic_data(data):
    """Generate a random sample of the input data.

    For each feature a random sample is taken of all objects.

    args:
        data:      input data with shape: objects x features
    returns:
        data_syn:  synthetic data same shape as input data
    """
    objects = data.shape[0]             # Amount of objects
    features = data.shape[1]            # Amount of features

    data_syn = np.zeros(data.shape)     # Prepare memory synthetic data

    # Generate a random sample for every feature with lenght of objects
    for i in range(features):
        data_syn[:, i] += np.random.choice(data[:, i], objects)

    return data_syn
    
    
def labeller(A, B):
    """Label input data

    Label input data with labels '1' and '2' for A and B respectively

    args:
        A:        data for label 1
        B:        data for label 2
    returns:
        data:     input data A and B combined
        labels:   labels corresponding to data output
    """
    # Build the labels vector
    A_label = np.ones(len(A))
    B_label = np.ones(len(B)) * 2

    # Pack the data and labels in two arrays
    data = np.concatenate((A, B))
    labels = np.concatenate((A_label, B_label))

    return data, labels
    
    
def merge_RF(RF, N_PROC=-1):

    # Add all trees to a single random forest
    for i in range(1, len(RF), 1):
        RF[0].estimators_ += RF[i].estimators_

    RF[0].n_estimators = len(RF[0].estimators_)
    RF[0].set_params(n_jobs=N_PROC)
    return RF[0]
    
    
def from_fits(path):
    with fits.open(path) as hdul:
        
        # Extract basic information
        hdr = hdul[0].header
        data = hdul[0].data
          
        # Compute wavelength range
        XMIN = hdr["CRVAL1"] - hdr["CD1_1"] * (hdr["CRPIX1"] - 1)
        XMAX = hdr["CRVAL1"] + hdr["CD1_1"] * (hdr["NAXIS1"] - hdr["CRPIX1"])
        X = np.linspace(XMIN, XMAX, hdr["NAXIS1"])

    return X, data[0], data[1], data[4], hdr
    
