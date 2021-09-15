# Kate Harborne 15/09/21
# Testing that EAGLE_cookies are cut out as expected.

# DISCLAIMER: These tests are being run independently by the user to check that 
#             properties like the periodic wrap and expected stellar mass 
#             content are pulled corrected from the simulation box. Due to the
#             large file sizes of the original simulation box, these tests are 
#             not automated using pytest. 

library(SimSpin)
library(hdf5r)
library(stringr)
source("~/repos/SimSpin/R/utilities.R")

# 1. The expected data is stored in the cutout details table:-------------------
EAGLE_galaxies = read.table("/mnt/858a2a9f-da71-4bdb-996d-882dbd918548/MAGPI/EAGLE/eagle_db_L0100N1504_snap28_aperture50kpc.txt",
                            sep= ",", header=T)

# 2. The galaxies that have been cut out at the moment include:-----------------
# These galaxies are cut out without galaxy ID information i.e. all particles
# in the sphere.
cutout_galaxies = list.files(path="/media/keaitch/One Touch/MAGPI/EAGLE/galaxy_cutouts/snap_28/")
cutout_ids      = as.integer(str_remove(str_remove(cutout_galaxies, "EAGLE_snap28_50kpc_galaxyID_"), ".hdf5"))
EAGLE_galaxies = EAGLE_galaxies[EAGLE_galaxies$GalaxyID %in% cutout_ids,]

# Getting header information
data = hdf5r::h5file("/media/keaitch/One Touch/MAGPI/EAGLE/galaxy_cutouts/snap_28/EAGLE_snap28_50kpc_galaxyID_8633758.hdf5", mode="r")
scale_factor = hdf5r::h5attr(data[["Header"]], "ExpansionFactor")
hubble_param = hdf5r::h5attr(data[["Header"]], "HubbleParam")
hdf5r::h5close(data)

# 3. Running tests:-------------------------------------------------------------
mass_diff  = numeric(length(cutout_galaxies)) # percentage difference in mass between measured and expected
avg_radius = numeric(length(cutout_galaxies)) # physical kpc between the middle and the max/min particle 
cop_x_diff = numeric(length(cutout_galaxies)) # percentage difference in COP between measured and expected
cop_y_diff = cop_x_diff; cop_z_diff = cop_x_diff

for (each in 1:length(cutout_galaxies)){
  # Checking that the total mass of the cutout galaxy matches the expected aperture mass
  galaxy = .read_hdf5(paste0("/media/keaitch/One Touch/MAGPI/EAGLE/galaxy_cutouts/snap_28/",  cutout_galaxies[each]))
  total_stellar_mass = sum(galaxy$star_part$Mass)
  mass_diff[each]  = ((EAGLE_galaxies$Mass_Star[each] - total_stellar_mass)/EAGLE_galaxies$Mass_Star[each])*100

  avg_radius[each] = mean(c((max(galaxy$star_part$x) - median(galaxy$star_part$x)),
                            (median(galaxy$star_part$x) - min(galaxy$star_part$x)),
                            (max(galaxy$star_part$y) - median(galaxy$star_part$y)),
                            (median(galaxy$star_part$y) - min(galaxy$star_part$y)),
                            (max(galaxy$star_part$z) - median(galaxy$star_part$z)),
                            (median(galaxy$star_part$z) - min(galaxy$star_part$z)))) # physical kpc
  
  cop_x_diff[each] = (((EAGLE_galaxies$CentreOfPotential_x[each] * scale_factor * 1e3) - median(galaxy$star_part$x)) / (EAGLE_galaxies$CentreOfPotential_x[each] * scale_factor * 1e3))*100
  cop_y_diff[each] = (((EAGLE_galaxies$CentreOfPotential_y[each] * scale_factor * 1e3) - median(galaxy$star_part$y)) / (EAGLE_galaxies$CentreOfPotential_y[each] * scale_factor * 1e3))*100
  cop_z_diff[each] = (((EAGLE_galaxies$CentreOfPotential_z[each] * scale_factor * 1e3) - median(galaxy$star_part$z)) / (EAGLE_galaxies$CentreOfPotential_z[each] * scale_factor * 1e3))*100 
}

# Negative differences imply that there is more stellar mass in the sphere than expected.
# This is to be expected, however, because the aperture values given in the EAGLE database
# are only for particles linked to that galaxy ID - while we have cut out everything within
# the radius. This implies that the difference must always be 0 or negative. 

# Are the the masses similar? And are they all negative or 0?
EAGLE_galaxies$GalaxyID[which(abs(mass_diff) > 1)] # larger numbers imply other galaxies in the sphere
EAGLE_galaxies$GalaxyID[which(mass_diff > 0)]

# Are the mass radii less than the requested region size? 
EAGLE_galaxies$GalaxyID[which(avg_radius > 50)] # hopefully this is always empty!

# Are the centres of the stelar distribution close to the COP?
EAGLE_galaxies$GalaxyID[which(abs(cop_x_diff) > 1)]
EAGLE_galaxies$GalaxyID[which(abs(cop_y_diff) > 1)]
EAGLE_galaxies$GalaxyID[which(abs(cop_z_diff) > 1)]