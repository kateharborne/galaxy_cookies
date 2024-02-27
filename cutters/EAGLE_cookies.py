# A script to pull out generic sized regions with or without GalaxyIDs
# from the EAGLE simulation.

import numpy as np
import pandas
import pyread_eagle as read_eagle
import gc
import h5py
import g3read

class CreateEagleGalaxyCutout:

    def __init__(self, first_eagle_file, centre, region_radius, gn, sgn):

        # read cosmological constants from first file header
        self.header = self.read_header(first_eagle_file)
        self.scale_factor = self.header["Time"]
        self.hubble_param = self.header["HubbleParam"]
        self.box_size     = self.header["BoxSize"]

        # load individual galaxy data
        self.data, self.region, self.numpart_total = self.read_galaxy(first_eagle_file, centre, region_radius, gn, sgn)


    def read_header(self, first_eagle_file):
        """
        Getting cosmological constants required to scale properties from comoving to physical.
        """
        self.header = {}

        f = h5py.File(first_eagle_file, "r")
        for attr in list(f["Header"].attrs):
            self.header[attr] = f["Header"].attrs.get(attr)

        self.header["RunLabel"] = "Eagle"

        f.close()

        return self.header

    def read_galaxy(self, first_eagle_file, centre, region_radius, gn, sgn):
        """
        Cutting out a spherical region about some "centre" position with radius "load_region_radius".
        first_eagle_file :: (string) the path to an eagle snapshot file fom the relevent box.
        centre :: (Numpy array) the [x,y,z] centres of potential of the individual galaxy in cMpc.
        region_radius :: (numeric) the radius of the enclosing sphere in physical kpc (must be smaller than 100kpc).
        gn :: (numeric) The group number of the galaxy.
        sgn :: (numeric) The subgroup number of the galaxy.
        """
        # Initiallising dictionary for all properties to be stored
        self.data={}

        # Put centre and boxsize into units of cMpc/h
        centre = centre * self.hubble_param
        boxsize = self.box_size * self.hubble_param

        # Select region to load
        assert region_radius < 100, "Requested region_radius must be less than 100kpc."

        self.region = np.array([ # selecting a region that is 100kpc big out of the box
            (centre[0]-0.1), (centre[0]+0.1),
            (centre[1]-0.1), (centre[1]+0.1),
            (centre[2]-0.1), (centre[2]+0.1)
        ])

        self.numpart_total = np.zeros(6, dtype="uint64")

        # Initialise the read_eagle module

        eagle_data = read_eagle.EagleSnapshot(first_eagle_file)
        eagle_data.select_region(*self.region)

        # Loop over particle types to process
        for itype in range(6):
            # Get number of particles to read
            nop = eagle_data.count_particles(itype)

            if nop > 0: # if there are particles in this particle type within the region
                coord = eagle_data.read_dataset(itype, "/Coordinates")
                coord[:,0] = (coord[:,0] - centre[0]) * (self.scale_factor/self.hubble_param) * 1e3 # in physical coordinates, kpc
                coord[:,1] = (coord[:,1] - centre[1]) * (self.scale_factor/self.hubble_param) * 1e3 # in physical coordinates, kpc
                coord[:,2] = (coord[:,2] - centre[2]) * (self.scale_factor/self.hubble_param) * 1e3 # in physical coordinates, kpc

                # masking out particles that lie outside of the required radius
                if gn is None: # if no gn and no sgn is provided, the cut out is just trimmed based on radius
                    mask = ((coord[:,0]*coord[:,0]) + (coord[:,1]*coord[:,1]) + (coord[:,2]*coord[:,2])) <= (region_radius*region_radius)
                else: # if gn and sgn are provided, the particles are further trimmed t only include relavent galaxy particles
                    mask1 = ((coord[:,0]*coord[:,0]) + (coord[:,1]*coord[:,1]) + (coord[:,2]*coord[:,2])) <= (region_radius*region_radius)
                    mask2 = np.logical_and(eagle_data.read_dataset(itype, '/GroupNumber') == gn, eagle_data.read_dataset(itype, '/SubGroupNumber') == sgn)
                    mask = np.logical_and(mask1, mask2)

                nopig = np.sum(mask) # number of particles within radius?

                if nopig > 0: # if there are particles of this type within the galaxy

                    print(f"Particle type {itype} - keeping {nopig} particles in galaxy from EAGLE snapshot file")
                    self.data[f"PartType{itype}"] = {}
                    read_datasets = eagle_data.datasets(itype)

                    for dset_name in read_datasets:
                        tmp = eagle_data.read_dataset(itype, dset_name)[mask]
                        self.data[f"PartType{itype}"][dset_name] = tmp


        del eagle_data

        for ptype in self.data.keys():
            # Periodic wrap coordiantes around centre
            self.data[f"{ptype}"]["/Coordinates"] = np.mod(self.data[f"{ptype}"]['/Coordinates'] - centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
            self.numpart_total[int(ptype[8])] = len(self.data[f"{ptype}"]["/ParticleIDs"])

        return self.data, self.region, self.numpart_total


def write_eagle_galaxy_to_file(galaxy_cutout, first_eagle_file, snap_num, output_location):
    """
    A function to accept a cutout galaxy object and write this information to an HDF5 file
    galaxy_cutout :: (Object) of the class CreateEagleGalaxyCutout
    output_location :: (String) describing the path to the HDF5 file written
    """

    f  = h5py.File(first_eagle_file, "r")
    hf = h5py.File(output_location, "w")    # creating HDF5 file for SimSpin

    header = hf.create_group("Header")
    for name in list(galaxy_cutout.header.keys()):
        header.attrs[name] = galaxy_cutout.header[name]

    # Update particle numbers in header
    nptot    = np.zeros(6, dtype="uint32")
    nptot_hw = np.zeros(6, dtype="uint32")
    nptot_hw[:] = galaxy_cutout.numpart_total >> 32
    nptot[:]    = galaxy_cutout.numpart_total - (nptot_hw << 32)
    header.attrs["NumPart_Total"] = nptot
    header.attrs["NumPart_Total_HighWord"] = nptot_hw
    header.attrs["NumPart_ThisFile"] = nptot

    # Now only have a single file
    header.attrs["NumFilesPerSnapshot"] = 1

    # Copy other groups with run information
    for group_name in ("Config",
                       "Constants",
                       "Parameters",
                       "Parameters/ChemicalElements",
                       "RuntimePars",
                       "Units"):
        group = hf.create_group(group_name)
        for (name,val) in f[group_name].attrs.items():
            group.attrs[name] = val

    header.attrs["ExtractedFromSnapshot"] = snap_num

    # Add region spec and type flags
    header.attrs["RegionExtracted"] = np.array(galaxy_cutout.region, dtype="float64")
    header.attrs["TypesExtracted"]  = np.array((1,1,1,1,1,1), dtype="int32")
    header.attrs["SamplingRate"] = 1.0

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
                for (name,val) in f[f"{ptype}/{att}"].attrs.items():
                    hf[f"{ptype}/{att}"].attrs[name] = val

    f.close()
    hf.close()

    print("Written galaxy to file: "+output_location)

def cutout_eagle_galaxies(first_eagle_file, snap_num, cutout_details, region_radius, output_location, galID = False):
    """
    A function to accept a table of GalaxyID/centres and produce HDF5 files for
    each galaxy contained in the table.

    first_eagle_file :: (String) describing the path to one of the eagle files
                        from the relevant snapshot
    snap_num         :: (Numeric) the snapshot number
    cutout_details   :: (String) describing the path to the GalaxyID/Centres table
    region_radius    :: (Numeric) the radius of the spherical region to be
                        extracted from the simulation (in physical kpc)
    output_location  :: (String) describing the path to the HDF5 file written
    galID            :: (Boolean) Default is False. Specify True if you would like to
                         only cut out particles associated to that GalaxyID.
    """

    regions_df = pandas.read_csv(cutout_details, comment="#")
    galaxy_no = len(regions_df.index)

    if galaxy_no > 1:

        for i in range(galaxy_no-1):

            if galID:
                gn = np.array([regions_df["GroupNumber"][i]])
                sgn = np.array([regions_df["SubGroupNumber"][i]])
            else:
                gn = None
                sgn = None

            centre = np.array([regions_df["CentreOfPotential_x"][i],
                               regions_df["CentreOfPotential_y"][i],
                               regions_df["CentreOfPotential_z"][i]])
            galaxy_cutout = CreateEagleGalaxyCutout(first_eagle_file = first_eagle_file, centre = centre, region_radius = region_radius, gn = gn, sgn = sgn)
            write_eagle_galaxy_to_file(galaxy_cutout = galaxy_cutout, first_eagle_file = first_eagle_file, snap_num = snap_num, output_location = output_location+str(regions_df["GalaxyID"][i])+".hdf5")
            print("Galaxy "+str(i+1)+" of "+str(galaxy_no-1))

    else:
        if galID:
            gn = np.array([regions_df["GroupNumber"]][0])
            sgn = np.array([regions_df["SubGroupNumber"]][0])
        else:
            gn = None
            sgn = None

        centre = np.array([regions_df["CentreOfPotential_x"][0],
                           regions_df["CentreOfPotential_y"][0],
                           regions_df["CentreOfPotential_z"][0]])
        galaxy_cutout = CreateEagleGalaxyCutout(first_eagle_file = first_eagle_file, centre = centre, region_radius = region_radius, gn = gn, sgn = sgn)
        write_eagle_galaxy_to_file(galaxy_cutout = galaxy_cutout, first_eagle_file = first_eagle_file, snap_num = snap_num, output_location = output_location+str(regions_df["GalaxyID"][0])+".hdf5")
        print("Galaxy "+str(1)+" of "+str(1))

