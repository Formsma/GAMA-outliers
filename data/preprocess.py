#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert individual GAMA spectra FITS files in to a single
array cut and sorted on signal to noise ratio

@author: Job Formsma
"""

import numpy as np                         
import scipy as sp
import pandas as pd                        
from astropy.io import fits                
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d     

from multiprocessing import Pool        
from tqdm import tqdm                   

import warnings                                 
warnings.filterwarnings("ignore")          

SN_cutoff = 2.5                    # S/N cutoff for all spectra, only 
                                   # spectra higher than input are used
kernel_size = 3                    # Gaussian convolution kernel

query_file = "GAMA_SpecAll.csv"    # CSV file with query data 
inp_path = "/fits/"                # Input path for FITS files
out_path = "/arrays/"              # Output path for files

N_processes = 64                   # Amount of processes to start

def main():
    
    spectra = pd.read_csv(query_file)           # Load GAMA data
    spectra = spectra[spectra.SN >= SN_cutoff]  # Signal/Noise cut
    spectra = spectra.sort_values(by="SN")      # Sort on SNR

    f = spectra.SPECID.values      # Extract unique spectra names

    N = len(f)                     # Subset size
    s = "_SN{:.0f}_{:.0f}".format(SN_cutoff, kernel_size)  # Added to output files

    if N != len(f):                # Check if subset has to be made
        i = np.random.choice(len(f), N, replace=False)
        f = f[np.sort(i)]
        
    # Exclude known SPECIDs
    known = pd.read_csv("GAMA_COMMENTS.csv")
    instrument = pd.read_csv("errors.csv")
    star = pd.read_csv("stars.csv")
    excludes = np.concatenate([known[known.COMMENT_FLAG >= 64].SPECID.values,
                               instrument.SPECID.values, 
                               star.SPECID.values])
    
    f = np.array([g for g in f if g not in excludes])
    N = len(f)
    print(N)
    
    # Save the specid names to a text file
    np.savetxt(out_path+"SPECS_{:.0f}{}.txt".format(N, s), f, fmt="%s")

    # Start multiprocess pool
    with Pool(processes=N_processes) as pool:
        storage = list(tqdm(pool.imap(worker, f), total=N, smoothing=0))
     
    # Write to memmap array
    map = np.memmap(out_path+"SPECS_{:.0f}{}.array".format(N, s), 
                    dtype='float16', mode='w+', 
                    shape=(len(storage), 8000))
    map[:] = storage
 
    # Also write to FITS file for easy visualisation
    hdu = fits.PrimaryHDU(storage)
    hdu.writeto(out_path+"SPECS_{:.0f}{}.fits".format(N, s))
   

def worker(SPECID):
    
    # Get data
    wave, flux, flux_error, sky, hdr = from_fits(SPECID)
    flux /= np.nanmedian(flux)
    
    # Mask sky line
    flux[np.where((wave > 5570) & (wave < 5585))[0]] = np.nan

    # Remove Extreme values
    flux[np.where((flux > 100) | (flux < -10))] = np.nan
    
    # Find neightbours of NaN values and mask those
    isnan = np.where(np.isnan(flux) | np.isnan(flux_error))[0]
    isnan = np.concatenate([isnan - 1, isnan, isnan + 1])[1:-1]
    flux[isnan] = np.nan
    
    # Interpolate NaN values inbetween points, probably very slow but neat trick
    flux = pd.DataFrame(flux).interpolate().values.ravel().tolist()

    # Smooth data
    kernel = Gaussian1DKernel(kernel_size)
    flux = convolve(flux, kernel)
    #flux = np.asarray(flux)
    #flux = medfilt(flux, 5)
    
    # Remove blue and red end
    flux[np.where((wave > 8786) | (wave < 4057))[0]] = np.nan
    
    # Redshift wavelengths
    wave /= 1 + hdr["Z"]
    
    # Interpolate flux values
    F = interp1d(wave, flux, bounds_error=False)
    wave_inp = np.linspace(3500, 7500, 8000)
    flux = F(wave_inp)
    
    # Remove NaN at boundaries
    isnan = np.isnan(flux)
    flux[isnan & (wave_inp < 4100)] = np.nanmedian(flux[~isnan][:100])
    flux[isnan & (wave_inp > 4100)] = np.nanmedian(flux[~isnan][-100:])
     
    # Normalize data to 1
    flux /= np.nanmedian(flux)
    
    # Remove last annoying values (often found in AGNs)
    flux[~np.isfinite(flux)] = 1
    
    return flux


def from_fits(specid):
    with fits.open(inp_path+specid+".fit") as hdul:
        
        # Extract basic information
        hdr = hdul[0].header
        data = hdul[0].data
          
        # Compute wavelength range
        XMIN = hdr["CRVAL1"] - hdr["CD1_1"] * (hdr["CRPIX1"] - 1)
        XMAX = hdr["CRVAL1"] + hdr["CD1_1"] * (hdr["NAXIS1"] - hdr["CRPIX1"])
        X = np.linspace(XMIN, XMAX, hdr["NAXIS1"])

    return X, data[0], data[1], data[4], hdr


if __name__ == "__main__":
    main()
