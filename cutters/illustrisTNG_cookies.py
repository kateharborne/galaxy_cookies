# A script to pull out generic sized regions with or without GalaxyIDs
# from the IllustrisTNG simulation.

import illustris_python as il
import h5py
import numpy as np
import pandas as pd
import gc
import os
import time
from multiprocessing import Pool

class CreateIllustrisTNGCutout:

    def __init__(self, base_path, snapshot, centre, region_radius, coord_df):

        # read cosmological constants from first file header
        illustris_file_loc = base_path+'snapdir_'+str(snapshot).zfill(3)
        self.header = read_header(illustris_file_loc+'/snap_'+str(snapshot).zfill(3)+'.0.hdf5')
        self.scale_factor = self.header["Time"]
        self.hubble_param = self.header["HubbleParam"]
        self.box_size     = self.header["BoxSize"]

        # load individual galaxy data
        self.data, self.region, self.numpart_total = self.read_region(base_path, centre, region_radius, snapshot, coord_df)

    def overlap(self, start1, end1, start2, end2):
        """Does the range (start1, end1) overlap with (start2, end2)?"""
        return end1 >= start2 and end2 >= start1

    def read_region(self, base_path, centre, region_radius, snapshot, coord_df=None):
        """
        Cutting out a spherical region about some "centre" position with radius "load_region_radius".
        base_path :: (string) the path to the directory containing the IllustrisTNG data (expected to 
            contain snapshot files in /snapdir_XXX and group files in /groups_XXX).
        centre :: (Numpy array) the [x,y,z] centres of potential of the individual galaxy in ckpc/h.
        region_radius :: (numeric) the radius of the enclosing sphere in physical kpc (must be smaller than 100kpc).
        coord_df :: (string) Path to the file containing the min/max of the coordinates in each 
            file chunk. Default is None, in which case the code to generate the txt file will be run. 
        """
        # Initiallising dictionary for all properties to be stored
        self.data={}

        # Select region to load
        assert region_radius < 100, "Requested region_radius must be less than 100kpc."

        region_radius = region_radius * self.header["Time"]**(-1) * self.header['HubbleParam'] # convert to comoving kpc

        self.region = np.array([ # selecting a region that is region-radius big out of the box
            (centre[0]-region_radius), (centre[0]+region_radius),
            (centre[1]-region_radius), (centre[1]+region_radius),
            (centre[2]-region_radius), (centre[2]+region_radius)
        ])

        self.numpart_total = np.zeros(6, dtype="uint64")

        # Check for coord_df and create if not provided
        if os.path.isfile(coord_df):
            coord_limits = pd.read_csv(coord_df)
        else:
            construct_indexes(file_loc = base_path+'snapdir_'+str(snapshot).zfill(3), snapshot = snapshot, coord_df = "coord_df.txt" )    
            coord_limits = pd.read_csv("coord_df.txt")
        
        coord_limits = coord_limits.assign(PullGas = 0, PullStars = 0)

        for chunk in range(len(coord_limits)):
            gas_x_overlap = self.overlap(start1 = coord_limits['GasMinX'][chunk], end1 = coord_limits['GasMaxX'][chunk],
                                    start2 = self.region[0], end2 = self.region[1])
            gas_y_overlap = self.overlap(start1 = coord_limits['GasMinY'][chunk], end1 = coord_limits['GasMaxY'][chunk],
                                    start2 = self.region[2], end2 = self.region[3])
            gas_z_overlap = self.overlap(start1 = coord_limits['GasMinZ'][chunk], end1 = coord_limits['GasMaxZ'][chunk],
                                    start2 = self.region[4], end2 = self.region[5])   

            coord_limits.loc[chunk, "PullGas"] = gas_x_overlap * gas_y_overlap * gas_z_overlap

            stars_x_overlap = self.overlap(start1 = coord_limits['StarMinX'][chunk], end1 = coord_limits['StarMaxX'][chunk],
                                    start2 = self.region[0], end2 = self.region[1])
            stars_y_overlap = self.overlap(start1 = coord_limits['StarMinY'][chunk], end1 = coord_limits['StarMaxY'][chunk],
                                    start2 = self.region[2], end2 = self.region[3])
            stars_z_overlap = self.overlap(start1 = coord_limits['StarMinZ'][chunk], end1 = coord_limits['StarMaxZ'][chunk],
                                    start2 = self.region[4], end2 = self.region[5])   

            coord_limits.loc[chunk, "PullStars"] = stars_x_overlap * stars_y_overlap * stars_z_overlap

        # which of the chunks should we open?
        coord_limits = coord_limits.assign(PullChunk = coord_limits['PullGas'] + coord_limits['PullStars'])

        # now step through each chunk and pull any associated particles

        for snap_chunk in range(len(coord_limits)):
            if coord_limits['PullChunk'][snap_chunk]:
                
                print('It is necessary to open chunk '+str(snap_chunk)+'...')
                chunk = self.read_chunk(coord_limits, base_path, snapshot, centre, region_radius, snap_chunk)

                if len(self.data.keys()) == 0:
                    self.data = chunk
                else:
                    for ptype in chunk.keys():
                        if ptype in self.data:
                            for each in list(chunk[f"{ptype}"].keys()):
                                self.data[f"{ptype}"][each] = np.append(self.data[f"{ptype}"][each], chunk[f"{ptype}"][each], axis = 0)
                        else: 
                            self.data[f"{ptype}"] = chunk[f"{ptype}"]

        print("Done reading chunks!")
        del chunk
        
        for ptype in self.data.keys():
            # Periodic wrap coordiantes around centre
            self.data[f"{ptype}"]["Coordinates"] = np.mod(self.data[f"{ptype}"]['Coordinates'] - centre+0.5*self.box_size, self.box_size) + centre-0.5*self.box_size
            self.numpart_total[int(ptype[8])] = len(self.data[f"{ptype}"]["ParticleIDs"])

        return self.data, self.region, self.numpart_total

    def read_chunk(self, coord_limits, base_path, snapshot, centre, region_radius, snap_chunk):
        
        start  = time.time()

        self.chunk = {}
                    
        f = h5py.File(base_path+'snapdir_'+str(snapshot).zfill(3)+'/snap_'+str(snapshot).zfill(3)+'.'+str(snap_chunk)+'.hdf5', "r")

        if coord_limits['PullGas'][snap_chunk] & coord_limits['PullStars'][snap_chunk]:
            print("Opening gas and stars...")

            gas_xyz = f["PartType0"]["Coordinates"][:]
            gas_xyz[:,0] = gas_xyz[:,0] - centre[0]
            gas_xyz[:,1] = gas_xyz[:,1] - centre[1]
            gas_xyz[:,2] = gas_xyz[:,2] - centre[2]

            star_xyz = f["PartType4"]["Coordinates"][:]
            star_xyz[:,0] = star_xyz[:,0] - centre[0]
            star_xyz[:,1] = star_xyz[:,1] - centre[1]
            star_xyz[:,2] = star_xyz[:,2] - centre[2]

            gas_mask = (gas_xyz[:,0]**2 + gas_xyz[:,1]**2 + gas_xyz[:,2]**2) <= (region_radius*region_radius)
            star_mask = (star_xyz[:,0]**2 + star_xyz[:,1]**2 + star_xyz[:,2]**2) <= (region_radius*region_radius)

            # Then open the file to pull both gas and stars
            self.chunk["PartType0"] = {}
            self.chunk["PartType4"] = {}

            for each in list(f["PartType0"].keys()):
                if np.ndim(f["PartType0"][each][:]) > 1:
                    self.chunk["PartType0"][each] = np.array(f["PartType0"][each][:])[gas_mask, :]
                else:
                    self.chunk["PartType0"][each] = np.array(f["PartType0"][each][:])[gas_mask]

            for each in list(f["PartType4"].keys()):
                if np.ndim(f["PartType4"][each][:]) > 1:
                    self.chunk["PartType4"][each] = np.array(f["PartType4"][each][:])[star_mask, :]
                else:
                    self.chunk["PartType4"][each] = np.array(f["PartType4"][each][:])[star_mask]

        else:
            if coord_limits['PullGas'][snap_chunk]:
                # Then just pull the gas
                gas_xyz = f["PartType0"]["Coordinates"][:]
                gas_xyz[:,0] = gas_xyz[:,0] - centre[0]
                gas_xyz[:,1] = gas_xyz[:,1] - centre[1]
                gas_xyz[:,2] = gas_xyz[:,2] - centre[2]
                gas_mask = (gas_xyz[:,0]**2 + gas_xyz[:,1]**2 + gas_xyz[:,2]**2) <= (region_radius*region_radius)
                print("Opening for gas...")

                self.chunk["PartType0"] = {}
                            
                for each in list(f["PartType0"].keys()):
                    if np.ndim(f["PartType0"][each][:]) > 1:
                        self.chunk["PartType0"][each] = np.array(f["PartType0"][each][:])[gas_mask, :]
                    else:
                        self.chunk["PartType0"][each] = np.array(f["PartType0"][each][:])[gas_mask]

            else:
                # Then just pull the stars
                star_xyz = f["PartType4"]["Coordinates"][:]
                star_xyz[:,0] = star_xyz[:,0] - centre[0]
                star_xyz[:,1] = star_xyz[:,1] - centre[1]
                star_xyz[:,2] = star_xyz[:,2] - centre[2]
                star_mask = (star_xyz[:,0]**2 + star_xyz[:,1]**2 + star_xyz[:,2]**2) <= (region_radius*region_radius)
                print("Opening for stars...")
                
                self.chunk["PartType4"] = {}
                            
                for each in list(f["PartType4"].keys()):
                    if np.ndim(f["PartType4"][each][:]) > 1:
                        self.chunk["PartType4"][each] = np.array(f["PartType4"][each][:])[star_mask, :]
                    else:
                        self.chunk["PartType4"][each] = np.array(f["PartType4"][each][:])[star_mask]
                    
        f.close()
        end     = time.time()   
        print(" Done with chunk "+str(snap_chunk)+" in "+str(end-start)+"!\n")
        
        return self.chunk

def read_header(first_sim_file):
    """
     Getting cosmological constants required to scale properties from comoving to physical.
    """
    header = {}
    
    f = h5py.File(first_sim_file, "r")
    
    for attr in list(f["Header"].attrs):
        header[attr] = f["Header"].attrs.get(attr)
    
    f.close()
     
    return header

def construct_indexes(file_loc, snapshot, coord_df):

    """
    Function to construct a matrix containing the min and max coordinates (x,y,z) in each of the IllustrisTNG
    snapshot files. Outputs recorded for gas and stars.
    file_loc :: (String) Path to the directory containing the snapshot chunks (not including file names).
    snapshot :: (Int) Snapshot number of the simulation chunks. Used to auto-populate the file name.
    coord_df :: (String) Path to output txt file containing the dataframe of min/max coordinates for each file chunk.
    """
    n_files = len(os.listdir(file_loc))

    coord_range = pd.DataFrame({
        'GasMinX':[0],
        "GasMaxX":[0],
        'GasMinY':[0],
        "GasMaxY":[0],
        'GasMinZ':[0],
        "GasMaxZ":[0],
        'StarMinX':[0],
        "StarMaxX":[0],
        'StarMinY':[0],
        "StarMaxY":[0],
        'StarMinZ':[0],
        "StarMaxZ":[0]
    })
    

    for snaps in range(n_files):
        f = h5py.File(file_loc+'/snap_'+str(snapshot).zfill(3)+'.'+str(snaps)+'.hdf5', "r")
        coord_range.loc[snaps] = [min(f["PartType0"]["Coordinates"][:, 0]), 
                                  max(f["PartType0"]["Coordinates"][:, 0]),
                                  min(f["PartType0"]["Coordinates"][:, 1]),
                                  max(f["PartType0"]["Coordinates"][:, 1]),
                                  min(f["PartType0"]["Coordinates"][:, 2]),
                                  max(f["PartType0"]["Coordinates"][:, 2]),
                                  min(f["PartType4"]["Coordinates"][:, 0]), 
                                  max(f["PartType4"]["Coordinates"][:, 0]),
                                  min(f["PartType4"]["Coordinates"][:, 1]),
                                  max(f["PartType4"]["Coordinates"][:, 1]),
                                  min(f["PartType4"]["Coordinates"][:, 2]),
                                  max(f["PartType4"]["Coordinates"][:, 2])]
        f.close()
        print("Done reading indexes of chunk "+ str(snaps)+"\n")

    pd.DataFrame.to_csv(coord_range, path_or_buf=coord_df, index_label="ChunkNum")  

def findGalaxies(base_path, snapshot, output, min_solar_mass = 1, max_halfmass_radius = 50):
    """
    Function to create a filtered list of galaxies within the snapshot that meet our requirements. 
    Output saved to csv. 
    base_path :: (Str) Path to where the simulation is stored. Assumes group files are in a folder within this location called groups_[snapshot].
    snapshot  :: (Int) Snapshot number beinsg studied.
    min_solar_mass :: (Numeric) Value of minimum solar mass considered for suitable galaxies in units of physical 1e10 Msolar.
    max_halfmass_radius :: (Numeric) Value of the maximum stellar half mass radius considered for suitable galaxies in units of physical kpc. 
    """
    # pull data on TNG's subfind subhalos (takes a long time)
    subhalos = il.groupcat.loadSubhalos(base_path,snapshot,fields=['SubhaloMassType', 'SubhaloHalfmassRadType', 'SubhaloPos'])

    header = read_header(first_sim_file=base_path+'snapdir_'+str(snapshot).zfill(3)+'/snap_'+str(snapshot).zfill(3)+'.0.hdf5')
    
    min_solar_mass = min_solar_mass * header['HubbleParam']
    max_halfmass_radius = max_halfmass_radius * header['HubbleParam'] / header["Time"]

    # add masses to matrix
    masses            = subhalos['SubhaloMassType']
    mass_df           = pd.DataFrame(masses)
    mass_df           = mass_df.drop([0,1,2,3,5], axis=1)
    mass_df.columns   = ['StellarMass']
    
    # add radiuses to matrix
    radii             = subhalos['SubhaloHalfmassRadType']
    radius_df         = pd.DataFrame(radii)
    radius_df         = radius_df.drop([0,1,2,3,5], axis=1)
    radius_df.columns = ['StellarHalfMassRadius']

    # new matrix with radius & mass
    df = mass_df.assign(StellarHalfMassRadius = radius_df)
    
    # drop all rows where galaxy does not meet requirements
    df = df[df["StellarMass"] > min_solar_mass]
    df = df[df["StellarHalfMassRadius"] < max_halfmass_radius]
    
    # reset index to 0,1,2,... index becomes a column & is renamed galaxy_ID
    df = df.reset_index()
    df = df.rename(columns={"index":"GalaxyID"})
    
    # create a new matrix with (co-moving & SimSpin) centres of galaxies
    centres           = subhalos['SubhaloPos'][df["GalaxyID"]]
    centre_df         = pd.DataFrame(centres)
    centre_df.columns = ['x','y','z']
    
    # add galaxy centre matrix to matrix with radius, mass & galaxy_ID
    df = df.assign(centre_x = centre_df['x'],
                   centre_y = centre_df['y'],
                   centre_z = centre_df['z'])

    pd.DataFrame.to_csv(df, path_or_buf=output, index=False)
        
    return df

def write_galaxy_to_file(galaxy_cutout, base_path, snapshot, output_location):
    """
    A function to accept a cutout galaxy object and write this information to an HDF5 file
    galaxy_cutout :: (Object) of the class CreateEagleGalaxyCutout
    output_location :: (String) describing the path to the HDF5 file written
    """

    f  = h5py.File(base_path+'snapdir_'+str(snapshot).zfill(3)+'/snap_'+str(snapshot).zfill(3)+'.0.hdf5', "r")
    hf = h5py.File(output_location, "a")    # creating HDF5 file for SimSpin

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
    header.attrs["RunLabel"] = "IllustrisTNG"

    # Now only have a single file
    header.attrs["NumFilesPerSnapshot"] = 1

    # Copy other groups with run information
    for group_name in ("Config",
                       "Parameters"):
        group = hf.create_group(group_name)
        for (name,val) in f[group_name].attrs.items():
            group.attrs[name] = val

    header.attrs["ExtractedFromSnapshot"] = snapshot

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
                if len(f[f"{ptype}/{att}"].attrs.keys()) == 0:
                    hf[f"{ptype}/{att}"].attrs["a_scaling"] = 0
                    hf[f"{ptype}/{att}"].attrs["h_scaling"] = 0
                    hf[f"{ptype}/{att}"].attrs["to_cgs"] = 1
                else:
                    for (name,val) in f[f"{ptype}/{att}"].attrs.items():
                        hf[f"{ptype}/{att}"].attrs[name] = val

    f.close()
    hf.close()

    print("Written galaxy to file: "+output_location)


def cutout_illustris_galaxies(base_path, snapshot, cutout_details, coordinate_chunks, region_radius, output_location, galID = False):
    """
    A function to accept a table of GalaxyID/centres and produce HDF5 files for
    each galaxy contained in the table.

    base_path        :: (String) describing the path to directory in which the TNG 
                        snapshot chunks are held
    snapshot         :: (Numeric) the snapshot number
    cutout_details   :: (String) describing the path to the GalaxyID/Centres table
    coordinate_chunks:: (String) path to file containing the coordinates in each hdf5 file
    region_radius    :: (Numersncric) the radius of the spherical region to be
                        extracted from the simulation (in physical kpc)
    output_location  :: (String) describing the path to the HDF5 file written
    galID            :: (Boolean) Default is False. Specify True if you would like to
                         only cut out particles associated to that GalaxyID.
    """

    regions_df = pd.read_csv(cutout_details, comment="#")
    galaxy_no = len(regions_df["GalaxyID"])
    
    if galaxy_no > 1: # loop through each galaxy in the list and construct a file

        for i in range(galaxy_no-1):

            if galID:
                coord_df = None
            else:
                coord_df = coordinate_chunks

            centre = np.array([regions_df["centre_x"][i],
                               regions_df["centre_y"][i],
                               regions_df["centre_z"][i]])
            galaxy_data = CreateIllustrisTNGCutout(base_path=base_path, snapshot=snapshot, centre=centre, region_radius=region_radius, coord_df=coord_df)
            write_galaxy_to_file(galaxy_cutout = galaxy_data, base_path = base_path, snapshot = snapshot, output_location = output_location+"/IllustrisTNG_snap"+str(snapshot)+"_GalaxyID"+str(regions_df["GalaxyID"][i])+".hdf5")
            print("Galaxy "+str(i+1)+" of "+str(galaxy_no)+"\n")

    else: # if just one galaxy is requested

        if galID:
            coord_df = None
        else:
            coord_df = coordinate_chunks
        
        centre = np.array([regions_df["centre_x"][0],
                           regions_df["centre_y"][0],
                           regions_df["centre_z"][0]])
        galaxy_data = CreateIllustrisTNGCutout(base_path=base_path, snapshot=snapshot, centre=centre, region_radius=region_radius, coord_df=coord_df)
        write_galaxy_to_file(galaxy_cutout = galaxy_data, base_path = base_path, snapshot = snapshot, output_location = output_location+"/IllustrisTNG_snap"+str(snapshot)+"_GalaxyID"+str(regions_df["GalaxyID"][0])+".hdf5")
        print("Galaxy "+str(1)+" of "+str(galaxy_no)+"\n")



