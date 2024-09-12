# A script to pull out generic sized regions with or without GalaxyIDs
# from the COLIBRE simulation.

import numpy as np
import pandas
import swiftsimio
import swiftgalaxy
import h5py
import unyt as u
from swiftsimio import SWIFTDataset

class CreateColibreGalaxyCutout:

    def __init__(self, virtual_snapshot_file, soap_catalogue_file, region_radius, gal_id):

        galaxy = swiftgalaxy.SWIFTGalaxy(virtual_snapshot_file, swiftgalaxy.SOAP(soap_catalogue_file, halo_index = gal_id))

        # read cosmological constants from first file header
        self.header = self.read_header(galaxy)
        self.scale_factor = self.header["Time"]
        self.hubble_param = self.header["HubbleParam"]
        self.box_size     = self.header["BoxSize"]

        # load individual galaxy data
        self.data, self.numpart_total = self.read_galaxy(galaxy, region_radius)

    def read_header(self, galaxy):
        """
        Getting cosmological constants required to scale properties from comoving to physical.
        """
        self.header = {}

        self.header["BoxSize"] = galaxy.stars.metadata.boxsize
        self.header["Redshift"] = galaxy.stars.metadata.redshift
        self.header["HubbleParam"] = galaxy.stars.metadata.cosmology.h
        self.header["MassTable"] = galaxy.stars.metadata.header['InitialMassTable']
        self.header["NumPart_Total"] = ([galaxy.gas.coordinates.shape[0], galaxy.dark_matter.coordinates.shape[0], 0, 0, galaxy.stars.coordinates.shape[0],0])
        self.header["RunLabel"] = "Colibre"

        return self.header
    
    def read_galaxy(self, galaxy, region_radius):
        """
        Cutting out a spherical region about some "centre" position with radius "load_region_radius".
        galaxy :: (string) the path to an eagle snapshot file fom the relevent box.
        region_radius :: (numeric) the radius of the enclosing sphere in physical kpc (must be smaller than 100kpc).
        """
        # Initiallising dictionary for all properties to be stored
        self.data={}

        ngas = galaxy.gas.coordinates.shape[0]
        nstars = galaxy.stars.coordinates.shape[0]
        ndm = galaxy.dark_matter.coordinates.shape[0]
   
        self.numpart_total = [ngas, ndm, 0, 0, nstars, 0]

        gas_attr = ["coordinates", "densities", "masses", "particle_ids", "metal_mass_fractions", "star_formation_rates", 
                    "smoothing_lengths", "velocities", "element_mass_fractions.carbon", "element_mass_fractions.oxygen",
                    "element_mass_fractions.hydrogen", "internal_energies", "temperatures"]
        gas_attr_name = ["Coordinates", "Density", "Mass", "ParticleIDs", "Metallicity", "StarFormationRate", "SmoothingLength",
                         "Velocity", "ElementAbundance/Carbon", "ElementAbundance/Oxygen", "ElementAbundance/Hydrogen",
                          "InternalEnergy", "Temperature" ]

        dm_attr = ["coordinates", "masses", "particle_ids", "velocities"]
        dm_attr_name = ["Coordinates", "Mass", "ParticleIDs", "Velocity"]

        star_attr = ["coordinates", "initial_masses", "masses", "particle_ids", "metal_mass_fractions", "velocities", "birth_scale_factors"]
        star_attr_name = ["Coordinates", "InitialMass", "Mass", "ParticleIDs", "Metallicity", "Velocity", "StellarFormationTime"]
    
        type_index = ["0","1","4"] 

        # Loop over particle types to process
        for num, ptype in enumerate(["gas", "dark_matter", "stars"]):
            # Get number of particles to read
            nop = getattr(galaxy, ptype).coordinates.shape[0]

            if nop > 0: # if there are particles in this particle type within the region

                coord = getattr(galaxy,ptype).coordinates.to_physical().to_value(u.kpc)       
                mask = (coord[:,0]**2 + coord[:,1]**2 + coord[:,2]**2) <= (region_radius*region_radius)        
                nopig = np.sum(mask)

                if nopig > 0: # if there are particles of this type within the galaxy radius

                    print(f"Particle type {type_index[num]} - keeping {nopig} particles in galaxy from Colibre snapshot file")
                    self.data[f"PartType{type_index[num]}"] = {}
                    
                    if ptype == "gas":
                        for anum, dset_name in enumerate(gas_attr):
                            tmp = getattr(getattr(galaxy,ptype),dset_name)[mask]
                            self.data[f"PartType{type_index[num]}"][gas_attr_name[anum]] = tmp

                    if ptype == "dark_matter":
                        for anum, dset_name in enumerate(dm_attr):
                            tmp = getattr(getattr(galaxy,ptype),dset_name)[mask]
                            self.data[f"PartType{type_index[num]}"][dm_attr_name[anum]] = tmp

                    if ptype == "stars":
                        for anum, dset_name in enumerate(star_attr):
                            tmp = getattr(getattr(galaxy,ptype),dset_name)[mask]
                            self.data[f"PartType{type_index[num]}"][star_attr_name[anum]] = tmp

        return self.data, self.numpart_total, self.header

def write_colibre_galaxy_to_file(galaxy_cutout, virtual_snapshot_file, snap_num, output_location):
    """
    A function to accept a cutout galaxy object and write this information to an HDF5 file
    galaxy_cutout :: (Object) of the class CreateColibreGalaxyCutout
    output_location :: (String) describing the path to the HDF5 file written
    """
    hf = h5py.File(output_location, "w")    # creating HDF5 file for SimSpin
    f = h5py.File(virtual_snapshot_file, "r")

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
    header.attrs["RegionExtracted"] = np.array(galaxy_cutout.data.centre.to_physical.to_value(u.kpc), dtype="float64")
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

def cutout_colibre_galaxies(first_colibre_file, snap_num, cutout_details, region_radius, output_location, galID = True):
    """
    A function to accept a table of GalaxyID/centres and produce HDF5 files for
    each galaxy contained in the table.

    first_colibre_file :: (String) describing the path to one of the colibre files
                           from the relevant snapshot
    snap_num         :: (Numeric) the snapshot number
    cutout_details   :: (String) describing the path to the GalaxyID/Centres table
    region_radius    :: (Numeric) the radius of the spherical region to be
                        extracted from the simulation (in physical kpc)
    output_location  :: (String) describing the path to the HDF5 file written
    galID            :: (Boolean) Default is True. Only cutting out particles
                         associated to that GalaxyID.
    """

    snap_num_text = str(snap_num).rjust(4,'0')
    virtual_snapshot_file = f"{first_colibre_file}/SOAP/colibre_with_SOAP_membership_{snap_num_text}.hdf5"
    soap_catalogue_file = f"{first_colibre_file}/SOAP/halo_properties_{snap_num_text}.hdf5"

    regions_df = pandas.read_csv(cutout_details, comment="#")
    galaxy_no = len(regions_df.index)

    if galaxy_no > 1:

        for i in range(galaxy_no-1):

            if galID:
                gal_id = regions_df["GalaxyID"][i]

            galaxy_cutout = CreateColibreGalaxyCutout(virtual_snapshot_file = virtual_snapshot_file, soap_catalogue_file=soap_catalogue_file, region_radius=region_radius, gal_id=gal_id)
            write_colibre_galaxy_to_file(galaxy_cutout = galaxy_cutout, virtual_snapshot_file = virtual_snapshot_file, snap_num = snap_num, output_location = output_location+str(regions_df["GalaxyID"][i])+".hdf5")
            print("Galaxy "+str(i+1)+" of "+str(galaxy_no-1))

    else:
        
        if galID:
            gal_id = regions_df["GalaxyID"][0]    

        galaxy_cutout = CreateColibreGalaxyCutout(virtual_snapshot_file=virtual_snapshot_file, soap_catalogue_file=soap_catalogue_file, region_radius=region_radius, gal_id=gal_id)
        write_colibre_galaxy_to_file(galaxy_cutout = galaxy_cutout, virtual_snapshot_file = virtual_snapshot_file, snap_num = snap_num, output_location =  output_location+str(regions_df["GalaxyID"][0])+".hdf5")
        print("Galaxy "+str(1)+" of "+str(1))

