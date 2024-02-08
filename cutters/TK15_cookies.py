# A script to pull out generic sized regions without GalaxyIDs
# from the TK15 simulation.

import pynbody
import pandas
import numpy as np
import gc
import h5py

class CreateTK15GalaxyCutout:

    def __init__(self, tk15_file_loc, centre, region_radius):

        # read cosmological constants from first file header
        self.header = self.read_header(tk15_file_loc)
        self.scale_factor = self.header["Time"]
        self.hubble_param = self.header["HubbleParam"]
        self.box_size     = self.header["BoxSize"]

        # load individual galaxy data
        self.data, self.numpart_total = self.read_galaxy(tk15_file_loc, centre, region_radius)


    def read_header(self, tk15_file_loc):
        """
        Getting cosmological constants required to scale properties from comoving to physical.
        """
        self.header = {}

        f = g3read.GadgetFile(tk15_file_loc)
        self.header["Npart"] = f.header.npart
        self.header["MassTable"] = f.header.mass
        self.header["Flag_Cooling"] = f.header.flag_cooling
        self.header["Flag_DoublePrecision"] = f.header.flag_doubleprecision
        self.header["Flag_Feedback"] = f.header.flag_feedback
        self.header["Flag_IC_Info"] = f.header.flag_ic_info
        self.header["Flag_Metals"] = f.header.flag_metals
        self.header["Flag_Sfr"] = f.header.flag_sfr
        self.header["Flag_StellarAge"] = f.header.flag_stellarage
        self.header["ExpansionFactor"] = f.header.time
        self.header["NumFilesPerSnapshot"] = f.header.num_files
        self.header["NumPart_ThisFile"] = f.header.npart
        self.header["NumPart_Total"] = f.header.npartTotal
        self.header["NumPart_Total_HighWord"] = f.header.NallHW
        self.header["Omega0"] = f.header.Omega0
        self.header["OmegaLambda"] = f.header.OmegaLambda
        self.header["Time"] = f.header.time
        self.header["Redshift"] = f.header.redshift
        self.header["HubbleParam"] = f.header.HubbleParam
        self.header["BoxSize"] = f.header.BoxSize
        self.header["RunLabel"] = "Magneticum"

        return self.header
