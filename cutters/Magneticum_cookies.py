# A script to pull out generic sized regions with or without GalaxyIDs
# from the Magneticum simulation.

import g3read
import pandas
import numpy as np
import gc
import h5py
import pickle

class CreateMagneticumGalaxyCutout:

    def __init__(self, magnet_file_loc, centre, region_radius, with_ids):

        # read cosmological constants from first file header
        self.header = self.read_header(magnet_file_loc+str('.0'))
        self.scale_factor = self.header["Time"]
        self.hubble_param = self.header["HubbleParam"]
        self.box_size     = self.header["BoxSize"]

        # load individual galaxy data
        self.data, self.numpart_total = self.read_galaxy(magnet_file_loc, centre, region_radius, with_ids)


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
        self.header["RunLabel"] = "Magneticum"

        return self.header

    def convert_dataset_names(self, dataset):
        """
        Given a list of strings of Gadget2 binary block names, convert to new HDF5 Group names
        """
        dataset = [item.replace('POS ', 'Coordinates') for item in dataset]
        dataset = [item.replace('MASS', 'Mass') for item in dataset]
        dataset = [item.replace('VEL ', 'Velocity') for item in dataset]
        dataset = [item.replace('U   ', 'InternalEnergy') for item in dataset]
        dataset = [item.replace('RHO ', 'Density') for item in dataset]
        dataset = [item.replace('SFR ', 'StarFormationRate') for item in dataset]
        dataset = [item.replace('AGE ', 'StellarFormationTime') for item in dataset]
        dataset = [item.replace('iM  ', 'InitialMass') for item in dataset]
        dataset = [item.replace('TEMP', 'Temperature') for item in dataset]
        dataset = [item.replace('Zs  ', 'Metallicity') for item in dataset]
        dataset = [item.replace('HSML', 'SmoothingLength_gas') for item in dataset]
        dataset = [item.replace('HSMS', 'SmoothingLength_stars') for item in dataset]
        dataset = [item.replace('ID  ', 'ParticleIDs') for item in dataset]
        dataset = [item.replace('POT ', 'Potential') for item in dataset]
        dataset = [item.replace('BHMA', 'BH_Mass') for item in dataset]
        dataset = [item.replace('BHMD', 'BH_Mdot') for item in dataset]
        dataset = [item.replace('BHPC', 'BH_ProgenitorCount') for item in dataset]
        dataset = [item.replace('ACRB', 'BH_SmoothingLength') for item in dataset]
        dataset = [item.replace('PHID', 'dPotential_dt') for item in dataset]
        dataset = [item.replace('VRMS', 'Gas_RMSVelocity') for item in dataset]
        dataset = [item.replace('VBLK', 'Gas_BulkVelocity') for item in dataset]
        dataset = [item.replace('ABVC', 'Artificial_Gas_BulkVelocity') for item in dataset]
        dataset = [item.replace('HOTT', 'MaximumTemperature') for item in dataset]
        dataset = [item.replace('CLDX', 'ColdGasFraction') for item in dataset]
        dataset = [item.replace('TNGB', 'TrueNeighbours') for item in dataset]
        dataset = [item.replace('NE  ', 'NumDensElectrons') for item in dataset]
        dataset = [item.replace('NH  ', 'NumDensNeutralH') for item in dataset]

        return dataset

    def get_attributes(self, dataset):
        """
        Given a list of HDF5 Group names, output a dictionary containing the relevant attributes
        for that HDF5 Group.
        """
        attributes = {'Coordinates':     {'CGSConversionFactor': 3.086e21,
                                          'h-scale-exponent': -1.0,
                                          'aexp-scale-exponent': 1.0,
                                          'VarDescription': 'Co-moving coordiantes [kpc/h]. Physical position: r = ax = Coordiantes h^-1 a CGSConversionFactor [cm]'},
                     'Mass':             {'CGSConversionFactor': 1.989e43,
                                          'h-scale-exponent': -1.0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Co-moving particle mass [1e10 Msol/h]. Physical masses: M = Mass h^-1 CGSConversionFactor [g]'},
                     'Velocity':         {'CGSConversionFactor': 1e5,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0.5,
                                          'VarDescription': 'Co-moving velocities [km/s /h]. Physical velocities: v = a dx/dt = Velocities a^1/2 CGSConversionFactor [cm/s]'},
                     'InternalEnergy':   {'CGSConversionFactor': 1e10,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Internal energy per unit mass  [(km/s)^2]. U CGSConversionFactor = u [(cm/s)^2]'},
                     'Density':          {'CGSConversionFactor': 6.767792e-22,
                                          'h-scale-exponent': 2.0,
                                          'aexp-scale-exponent': -3.0,
                                          'VarDescription': 'Comoving mass density [1e10 Msol/h / (kpc/h)^3]. Rho = Density h^2 a^-3 CGSConversionFactor [g/(cm)^3]'},
                     'StarFormationRate':{'CGSConversionFactor': 6.306278e35,
                                          'h-scale-exponent': -1.0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Star formation rate [1e10 Msol/h / yr]. SFR_cgs = SFR h^-1 CGSConversionFactor [g/s]'},
                     'StellarFormationTime':{'CGSConversionFactor': 1.0,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Expansion factor, a, when star formed.'},
                     'InitialMass':      {'CGSConversionFactor': 1.989e43,
                                          'h-scale-exponent': -1.0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Co-moving particle mass [1e10 Msol/h]. Physical masses: M = Mass h^-1 CGSConversionFactor [g]'},
                     'Temperature':      {'CGSConversionFactor': 1.0,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Temperature [K]'},
                     'Metallicity':      {'CGSConversionFactor': 1.989e43,
                                          'h-scale-exponent': -1.0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Vector containing the mass of each 11 elements [1e10 Msol/h] He, C, Ca, O, N, Ne, Mg, S, Si, Fe, Eh (remaining metals). Physical masses: M = Mass h^-1 CGSConversionFactor [g]'},
                     'SmoothingLength_gas': {'CGSConversionFactor': 3.086e21,
                                             'h-scale-exponent': -1.0,
                                             'aexp-scale-exponent': 1.0,
                                             'VarDescription': 'Co-moving smoothing lengths [kpc/h]. Physical smoothing lengths: r = ax = Coordiantes h^-1 a CGSConversionFactor [cm]'},
                     'SmoothingLength_stars': {'CGSConversionFactor': 3.086e21,
                                               'h-scale-exponent': -1.0,
                                               'aexp-scale-exponent': 1.0,
                                               'VarDescription': 'Co-moving smoothing lengths [kpc/h]. Physical smoothing lengths: r = ax = Coordiantes h^-1 a CGSConversionFactor [cm]'},
                     'ParticleIDs':     {'CGSConversionFactor': 1.0,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': 0,
                                         'VarDescription': 'Unique particle identifier'},
                     'Potential':       {'CGSConversionFactor': 6.445237e+21,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': -1,
                                         'VarDescription': 'Co-moving gravitational potential energy [1e10 Msol / kpc]. V = Potential a^-1 CGSConversionFactor [g/cm]'},
                     'BH_Mass':         {'CGSConversionFactor': 1.989e43,
                                         'h-scale-exponent': -1.0,
                                         'aexp-scale-exponent': 0,
                                         'VarDescription': 'Co-moving black hole particle mass [1e10 Msol/h]. Physical masses: M = Mass h^-1 CGSConversionFactor [g]'},
                     'BH_Mdot':         {'CGSConversionFactor': 6.306278e35,
                                         'h-scale-exponent': -1.0,
                                         'aexp-scale-exponent': 0,
                                         'VarDescription': 'Co-moving black hole accretion rate [1e10 Msol/h / yr]. Physical rate: M = Mass h^-1 CGSConversionFactor [g/s]'},
                     'BH_ProgenitorCount':{'CGSConversionFactor': 1.0,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': 0,
                                         'VarDescription': 'Progenitor count of balck holes'},
                     'BH_SmoothingLength':{'CGSConversionFactor': 3.086e21,
                                         'h-scale-exponent': -1.0,
                                         'aexp-scale-exponent': 1.0,
                                         'VarDescription': 'Co-moving black hole smoothing lengths [kpc/h]. Physical smoothing lengths: r = ax = Coordiantes h^-1 a CGSConversionFactor [cm]'},
                     'dPotential_dt':   {'CGSConversionFactor': 2.043512e+14,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': -1,
                                         'VarDescription': 'Co-moving gravitational potential energy per unit time [1e10 Msol / kpc / yr]. dU/dt = dPotential/dt a^-1 CGSConversionFactor [g/cm/s]'},
                     'Gas_RMSVelocity': {'CGSConversionFactor': 1e5,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': 0.5,
                                         'VarDescription': 'Co-moving gas RMS velocities within kernel [km/s /h]. Physical velocities: v = a dx/dt = Velocities a^1/2 CGSConversionFactor [cm/s]'},
                     'Gas_BulkVelocity':{'CGSConversionFactor': 1e5,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': 0.5,
                                         'VarDescription': 'Co-moving, mean bulk gas velocities within kernel [km/s /h]. Physical velocities: v = a dx/dt = Velocities a^1/2 CGSConversionFactor [cm/s]'},
                     'Artificial_Gas_BulkVelocity':{'CGSConversionFactor': 1e5,
                                         'h-scale-exponent': 0,
                                         'aexp-scale-exponent': 0.5,
                                         'VarDescription': 'Co-moving, mean bulk gas velocities within kernel [km/s /h]. Physical velocities: v = a dx/dt = Velocities a^1/2 CGSConversionFactor [cm/s]'},
                     'MaximumTemperature':{'CGSConversionFactor': 1.0,
                                           'h-scale-exponent': 0,
                                           'aexp-scale-exponent': 0,
                                           'VarDescription': 'Maximum temperature ever reached by particle [K]'},
                     'ColdGasFraction':  {'CGSConversionFactor': 1.0,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Cold gas fraction from Springel & Hernquist model.'},
                     'TrueNeighbours':   {'CGSConversionFactor': 1.0,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'True number of neighbours.'},
                     'NumDensElectrons':{'CGSConversionFactor': 1.0,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Number density of free electrons.'},
                     'NumDensNeutralH':{'CGSConversionFactor': 1.0,
                                          'h-scale-exponent': 0,
                                          'aexp-scale-exponent': 0,
                                          'VarDescription': 'Number density of neutral hydrogen.'}

                    }

        output = {k: attributes[k] for k in dataset if k in attributes}

        return output

    def read_galaxy(self, magnet_file_loc, centre, region_radius, with_ids):
        """
        Cutting out a spherical region about some "centre" position with radius "load_region_radius".

        magnet_file_loc :: (string) the path to a magneticum snapshot files fom the relevent box
        centre :: (Numpy array) the [x,y,z] centres of potential of the individual galaxy
        region_radius :: (numeric) the radius of the enclosing sphere in physical kpc
        with_ids :: (Numpy array) the particle IDs of the particles associated with that particular galaxy

        """
        # Initiallising dictionary for all properties to be stored
        self.data={}

        # Put region radius into units of ckpc/h
        region_radius = region_radius #* (self.scale_factor/self.hubble_param)
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

                    if with_ids is None: #the cut out is just trimmed based on radius
                        # masking out particles that lie outside of the required radius
                        mask = ((coord[:,0]*coord[:,0]) + (coord[:,1]*coord[:,1]) + (coord[:,2]*coord[:,2])) < (region_radius*region_radius)
                    else:
                        mask1 = ((coord[:,0]*coord[:,0]) + (coord[:,1]*coord[:,1]) + (coord[:,2]*coord[:,2])) < (region_radius*region_radius)
                        mask2 = np.isin(magnet_data[itype]['ID  '], with_ids)
                        mask = np.logical_and(mask1, mask2)

                    nopig = np.sum(mask) # number of particles within radius?

                    if nopig > 0: # if there are particles of this type within the galaxy

                        print(f"Particle type {itype} - keeping {nopig} particles in galaxy from Magneticum snapshot file")
                        self.data[f"PartType{itype}"] = {}
                        read_datasets = list(magnet_data[itype].keys())
                        new_dataset_name = self.convert_dataset_names(read_datasets)
                        attrs = self.get_attributes(new_dataset_name)

                        for dset_name in read_datasets:
                            tmp = magnet_data[itype][dset_name][mask]
                            ds_index = read_datasets.index(dset_name)
                            self.data[f"PartType{itype}"][new_dataset_name[ds_index]] = {}
                            self.data[f"PartType{itype}"][new_dataset_name[ds_index]]['data'] = tmp
                            self.data[f"PartType{itype}"][new_dataset_name[ds_index]]['attrs'] = attrs[new_dataset_name[ds_index]]

        for ptype in self.data.keys():
            # Periodic wrap coordiantes around centre IS UNNECESSARY AS MAGNETICUM FUNCTION AUTO DOES THIS!
            #self.data[f"{ptype}"]['Coordinates']["data"] = np.mod(self.data[f"{ptype}"]['Coordinates']["data"] - centre+0.5*boxsize, boxsize) + centre-0.5*boxsize
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

def cutout_magneticum_galaxies(magnet_file_loc, snap_num, cutout_details, region_radius, output_location, galID):

    """
    A function to accept a table of GalaxyID/centres and produce HDF5 files for
    each galaxy contained in the table.

    magnet_file_loc  :: (String) describing the path to the magneticum files
                        from the relevent snapshot
    snap_num         :: (Numeric) the snapshot number
    cutout_details   :: (String) describing the path to the GalaxyID/Centres table
                        in numpy format.
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
        else:
            part_ids = None

        galaxy_cutout = CreateMagneticumGalaxyCutout(magnet_file_loc = magnet_file_loc, centre = centre, region_radius = region_radius, with_ids = part_ids)
        write_magneticum_galaxy_to_file(galaxy_cutout = galaxy_cutout, snap_num = snap_num, output_location = output_location+str(regions_df["subhalo_ID"][i])+".hdf5")
        print("Galaxy "+str(i + 1)+" of "+str(galaxy_no))
