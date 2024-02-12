# A script to write out RAMSES outputs into an HDF5 SimSpin readable format.

import numpy as np
import pynbody
import h5py

class CreateRamsesNbodyGalaxyCutout:

    def __init__(self, file_directory, centre, MaxCompNumber):
        
        # read cosmological constants from file header
        self.header = self.read_header(file_directory)
        self.scale_factor = self.header["a"]
        self.hubble_param = self.header["h"]
        self.box_size     = float(self.header["boxsize"])

        # load individual galaxy data
        self.data, self.nbody_flag, self.offset, self.numpart_total = self.read_galaxy(file_directory, centre, MaxCompNumber)

    def read_header(self, file_directory):
        """
        Getting cosmological constants required to scale properties from comoving to physical.
        """
        f = pynbody.load(file_directory)
        f.physical_units()
        self.header = f.properties

        return self.header

    def read_galaxy(self, file_directory, centre, MaxCompNumber):
        """
        Cutting out a region about some "centre" position with radius "load_region_radius".
        file_directory :: (string) the path to a RAMSES output directory.
        centre :: (Numpy array) the [x,y,z] centres of potential of the individual galaxy in kpc. If unsupplied, takes centre of the box. 
        MaxCompNumber :: (numeric) ?
        """
        f = pynbody.load(file_directory)
        f.physical_units()
        
        # Check to see if the input file is an Nbody model
        if len(f.families()) == 1: 
            self.nbody_flag = True
        else:
            assert self.nbody_flag, "Expecting a RAMSES N-body model."
        
        # Set centre to the centre of the simulation box if no centre position 
        # is supplied
        if centre == None:
            self.offset = self.box_size * 0.5
        else:
            self.offset = centre 
        
        
        # Using Nabo's method to sort dark matter particles into their respective "PartType"s
            
        self.data = {}
        self.numpart_total = [0, # number of gas particles
                             len(f.dm[ np.where(f.dm['iord'] == 1) ]), # number of DM particles
                             len(f.dm[ np.where(f.dm['iord'] == 3) ]), # number of bulge particles
                             len(f.dm[ np.where( (f.dm['iord'] % (2*MaxCompNumber)) == 5) ]), # number of disk particles
                             0,0] # number of stellar and boundary

        part = {}
        part["PartType0"] = np.nan
        part["PartType1"] = f.dm[ np.where(f.dm['iord'] == 1) ]
        part["PartType2"] = f.dm[ np.where(f.dm['iord'] == 3) ]
        part["PartType3"] = f.dm[ np.where( (f.dm['iord'] % (2*MaxCompNumber)) == 5) ]
        part["PartType4"] = np.nan
        part["PartType5"] = np.nan

        for itype in range(6):
            if self.numpart_total[itype] > 0:
                print(f"Particle type {itype} - keeping {self.numpart_total[itype]} particles in galaxy from RAMSES nbody file")
                self.data[f"PartType{itype}"] = {}
                self.data[f"PartType{itype}"]["ParticleIDs"] = part[f"PartType{itype}"]['iord']
                self.data[f"PartType{itype}"]["Coordinates"] = part[f"PartType{itype}"]['pos'] - self.offset
                self.data[f"PartType{itype}"]["Velocities"] = part[f"PartType{itype}"]['vel']
                self.data[f"PartType{itype}"]["Masses"] = part[f"PartType{itype}"]['mass'].in_units("1e10 Msol") # matching the Gadget2 unit style (Mass in 1e10 Msol)

        return self.data, self.nbody_flag, self.offset, self.numpart_total


def write_ramses_galaxy_to_file(galaxy_cutout, output_location):
    """
    A function to accept a cutout galaxy object and write this information to an HDF5 file

    galaxy_cutout :: (Object) of the class CreateGalaxyCutout
    output_location :: (String) describing the path to the HDF5 file written
    """

    hf = h5py.File(output_location, "w")    # creating HDF5 file for SimSpin

    header = hf.create_group("Header")
    header.attrs["Redshift"] = (1/galaxy_cutout.header["a"]) - 1
    header.attrs["Time"] = float(galaxy_cutout.header["time"])
    header.attrs["NumPart_ThisFile"] = galaxy_cutout.numpart_total

    for ptype in galaxy_cutout.data.keys():
        if galaxy_cutout.numpart_total[int(ptype[8])] > 0: # if there are any ptype particles associated with the galaxy
            for att in galaxy_cutout.data[ptype].keys():
                chunks = [s for s in galaxy_cutout.data[ptype][att].shape]
                chunks[0] = min((8192, chunks[0]))
                hf.create_dataset(f"{ptype}/{att}",
                                data=galaxy_cutout.data[ptype][att],
                                chunks=tuple(chunks),
                                shuffle=True,
                                compression="gzip",
                                compression_opts=6)

    hf.close()

    print("Written galaxy to file: "+output_location)

def cutout_ramses_nbody_galaxy(ramses_dir_loc, centre, max_comp_number, output_file):

    """
    A function to convert an input RAMSES n-body model to an HDF5 file
    ramses_dir_loc   :: (String) describing the path to the RAMSES output directory
    centre           :: (Numeric) Numpy array describing the (x,y,z) centre in kpc. 
                         If None, will centre based on the centre of the box.
    max_comp_number  :: (Numeric) a value that describes how to select disk particles?
    output_file      :: (String) describing the path and name of the HDF5 file written

    """

    galaxy_cutout = CreateRamsesNbodyGalaxyCutout(file_directory = ramses_dir_loc, centre = centre, MaxCompNumber = max_comp_number)
    write_ramses_galaxy_to_file(galaxy_cutout = galaxy_cutout, output_location = output_file)
    print("RAMSES galaxy written to HDF5 file successfully. \n")



def main():

    # Change these inputs to match your file locations
    ramses_dir_loc = "~/Dropbox/output_00001/"
    centre = None
    max_comp_number = 11
    output_file = "~/Dropbox/Files for SimSpin/output_00001.hdf5"

    cutout_ramses_nbody_galaxy(ramses_dir_loc, centre, max_comp_number, output_file)

if __name__ == "__main__":
    main()
