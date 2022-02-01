# A script to write out RAMSES outputs into an HDF5 SimSpin readable format.

import numpy as np
import pandas
import pyread_eagle as read_eagle
import gc
import h5py

def cutout_ramses_galaxies(file_directory, region_radius, output_location):

    # Get the scale length properties from the info_***.txt file
    with open(file_directory+str("/info_"))
