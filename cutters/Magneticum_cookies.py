# A script to pull out generic sized regions with or without GalaxyIDs
# from the Magneticum simulation.

import g3read
import g3read_units
import pandas
import numpy as np
import gc
import h5py

class CreateMagneticumGalaxyCutout:

    def __init__(self, magnet_file_loc, centre, region_radius):

        # read cosmological constants from first file header
        self.header = self.read_header(magnet_file_loc+str('.0'))
        self.scale_factor = self.header["Time"]
        self.hubble_param = self.header["HubbleParam"]
        self.box_size     = self.header["BoxSize"]

        # load individual galaxy data
        self.data, self.numpart_total = self.read_galaxy(magnet_file_loc, centre, region_radius)


    def read_header(self, first_magnet_file):
        """
        Getting cosmological constants required to scale properties from comoving to physical.
        """
        self.header = {}

        f = g3read.GadgetFile(first_magnet_file)
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

        return self.header

    def read_galaxy(self, magnet_file_loc, centre, region_radius):
        """
        Cutting out a spherical region about some "centre" position with radius "load_region_radius".

        magnet_file_loc :: (string) the path to a magneticum snapshot files fom the relevent box
        centre :: (Numpy array) the [x,y,z] centres of potential of the individual galaxy
        region_radius :: (numeric) the radius of the enclosing sphere in physical kpc

        """
        # Initiallising dictionary for all properties to be stored
        self.data={}

        # Put region radius into units of ckpc/h
        region_radius = region_radius * (self.scale_factor/self.hubble_param)
        boxsize = (self.box_size / self.scale_factor)

        self.numpart_total = np.zeros(6, dtype="uint64")

        magnet_data = g3read.read_particles_in_box(snap_file_name=magnet_file_loc, center=centre, d=region_radius,
                                                   blocks=('ID  ', 'POS ', 'VEL ', 'MASS', 'U   ', 'RHO ', 'NE  ', 'NH  ', 'HSML',
                                                           'SFR ', 'AGE ', 'HSMS', 'BHMA', 'BHMD', 'BHPC', 'ACRB', 'POT ',
                                                           'PHID', 'ABVC', 'VRMS', 'VBLK', 'TNGB', 'iM  ', 'Zs  ', 'CLDX',
                                                           'HOTT', 'TEMP'),
                                                   ptypes=-1, join_ptypes=False)

        # Loop over particle types to process
        for itype in range(6):

            if bool(magnet_data[itype]):
                # Get number of particles to read
                nop = len(magnet_data[itype]['ID  '])

                if nop > 0: # if there are particles in this particle type within the region
                    coord = magnet_data[itype]['POS ']
                    coord[:,0] = (coord[:,0] - centre[0]) * (self.scale_factor/self.hubble_param) # in physical coordinates, kpc
                    coord[:,1] = (coord[:,1] - centre[1]) * (self.scale_factor/self.hubble_param) # in physical coordinates, kpc
                    coord[:,2] = (coord[:,2] - centre[2]) * (self.scale_factor/self.hubble_param) # in physical coordinates, kpc
                    # masking out particles that lie outside of the required radius
                    mask = ((coord[:,0]*coord[:,0]) + (coord[:,1]*coord[:,1]) + (coord[:,2]*coord[:,2])) < (region_radius*region_radius)
                    nopig = np.sum(mask) # number of particles within radius?

                    if nopig > 0: # if there are particles of this type within the galaxy

                        print(f"Particle type {itype} - keeping {nopig} particles in galaxy from Magneticum snapshot file")
                        self.data[f"PartType{itype}"] = {}
                        read_datasets = list(magnet_data[itype].keys())
                        new_dataset_name = convert_dataset_names(read_datasets)
                        attrs = get_attributes(new_dataset_name)

                        for dset_name in read_datasets:
                            tmp = magnet_data[itype][dset_name][mask]
                            ds_index = read_datasets.index(dset_name)
                            self.data[f"PartType{itype}"][new_dataset_name[ds_index]] = {}
                            self.data[f"PartType{itype}"][new_dataset_name[ds_index]]['data'] = tmp
                            self.data[f"PartType{itype}"][new_dataset_name[ds_index]]['attrs'] = attrs[new_dataset_name[ds_index]]

        for ptype in self.data.keys():
            # Periodic wrap coordiantes around centre

            self.data[f"{ptype}"]['Coordinates']["data"] = np.mod(self.data[f"{ptype}"]['Coordinates']["data"] - centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
            self.numpart_total[int(ptype[8])] = len(self.data[f"{ptype}"]['ParticleIDs']['data'])

        return self.data, self.numpart_total

def write_magneticum_galaxy_to_file(galaxy_cutout, snap_num, output_location):
    """
    A function to accept a cutout galaxy object and write this information to an HDF5 file

    galaxy_cutout :: (Object) of the class CreateGalaxyCutout
    output_location :: (String) describing the path to the HDF5 file written
    """

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
    header.attrs["ExtractedFromSnapshot"] = snap_num

    # Add region spec and type flags
    header.attrs["TypesExtracted"]  = np.array((1,1,1,1,1,1), dtype="int32")
    header.attrs["SamplingRate"] = 1.0

    for ptype in galaxy_cutout.data.keys():
        if galaxy_cutout.numpart_total[int(ptype[8])] > 0: # if there are any ptype particles associated with the galaxy
            for att in galaxy_cutout.data[ptype].keys():
                chunks = [s for s in galaxy_cutout.data[ptype][att]['data'].shape]
                chunks[0] = min((8192, chunks[0]))
                hf.create_dataset(f"{ptype}/{att}",
                                data=galaxy_cutout.data[ptype][att]['data'],
                                chunks=tuple(chunks),
                                shuffle=True,
                                compression="gzip",
                                compression_opts=6)
                for key in galaxy_cutout.data[ptype][att]['attrs']:
                    hf[f"{ptype}/{att}"].attrs[key] = galaxy_cutout.data[ptype][att]['attrs'][key]

    hf.close()

    print("Written galaxy to file: "+output_location)

def cutout_magneticum_galaxies(magnet_file_loc, snap_num, regions_file, region_radius, output_location):
    """
    A function to accept a table of GalaxyID/centres and produce HDF5 files for
    each galaxy contained in the table.

    magnet_file_loc :: (String) describing the path to the magneticum files
                        from the relevent snapshot
    snap_num         :: (Numeric) the snapshot number
    cutout_details   :: (String) describing the path to the GalaxyID/Centres table
    region_radius    :: (Numeric) the radius of the spherical region to be
                        extracted from the simulation
    output_location  :: (String) describing the path to the HDF5 file written
    galID            :: (Boolean) Default is False. Specify True if you would like to
                         only cut out particles associated to that GalaxyID.
    """

    with open(cutout_details, "rb") as fp:
        regions_df = pickle.load(fp)

    galaxy_no = len(regions_df["subhalo_ID"])
    for i in range(galaxy_no):
        centre = np.array([regions_df["subhalo_centre_x"][i],
                           regions_df["subhalo_centre_y"][i],
                           regions_df["subhalo_centre_z"][i]])
        if galID:
            part_ids = np.array([regions_df["particle_ids"][i]])

        galaxy_cutout = CreateMagneticumGalaxyCutout(magnet_file_loc = magnet_file_loc, centre = centre, region_radius = region_radius, with_ids = part_ids)
        write_galaxy_to_file(galaxy_cutout = galaxy_cutout, snap_num = snap_num, output_location = output_location+str(regions_df["subhalo_ID"][i])+".hdf5")
        print("Galaxy "+str(i + 1)+" of "+str(galaxy_no))
