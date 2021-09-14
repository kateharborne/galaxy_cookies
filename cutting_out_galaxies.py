import os
from argparse import ArgumentParser
from datetime import date
from cutters.EAGLE_cookies import *

# Define arguments ----
parser = ArgumentParser()
parser.add_argument("-t", "--type", action="store", dest="sim_type", type=str,
                    default=False, help="Simulation being cut out. Options include EAGLE and Magneticum.")
parser.add_argument("-f", "--first_file", action="store", dest="first_file", type=str,
                    default=None, help="The path to one of the files from the relevant snapshot.")
parser.add_argument("-c", "--cutout_details", action="store", dest="cutout_details", type=str,
                    default=None, help="The path to the GalaxyID/Centres table.")
parser.add_argument("-o", "--output_loc", action="store", dest="output_location", type=str,
                    default=None, help="The path to the directory at which HDF5 files will be written.")
parser.add_argument("-s", "--snap_num", action="store", dest="snap_num", type=float,
                    default=None, help="The snapshot number from which galaxies are cut.")
parser.add_argument("-r", "--radius", action="store", dest="region_radius", type=float,
                    default=None, help="The radius of the spherical region to be extracted from the simulation (in physical kpc).")
parser.add_argument("-g", "--galID", action="store", dest="galID", type=Boolean,
                    default=False, help="Specify True if you would like to only cut out particles associated to that GalaxyID.")

# Parse arguments ----
args = parser.parse_args()

if args.sim_type == "EAGLE" or args.sim_type == "eagle":
    out_files = f"{args.output_loc}/{args.sim_type}_snap{args.snap_num}_{args.region_radius}kpc_galaxyID_"
    cutout_eagle_galaxies(first_eagle_file = args.first_file, snap_num = args.snap_num, cutout_details = args.cutout_details, region_radius = args.region_radius, output_location = out_files, galID=args.galID)

    with open(f"{args.output_loc}/README.txt", 'w') as f:
        f.write('Summary of galaxy cutout files \n')
        f.write(f"Written on {date.today()} \n")
        f.write("\n")
        f.write(f"Simulation: {args.sim_type}\n")
        f.write(f"Centres and galaxy IDs specified by the table: {args.cutout_details}\n")
        f.write(f"Files contain particles within spherical radius: {args.region_radius} kpc\n")
        f.write(f"Files contain just particles associated with the galaxy ID?: {args.galID}\n")
