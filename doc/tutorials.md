# Tutorials

## Assumptions

These tutorials assume that you are familiar with the basic operations and
common calculation setups in ASE, i.e., loading the structure from cif or VASP
POSCAR, instantiating calculators and assigning them to the loaded structures,
calculating the total energy and forces by calling the 'get_potential_energy'
and 'get_forces' methods, and optimizing the structure using subroutines in
ase.optimize package. It is also assumed that you know the basics of the 
application of machine learning in predicting the potential energy surface,
especially the multilayer feedforward neuron network.

## Overview

We take the diffusion of an Au atom on Au001 surface for example as it takes
less computational resources. We will calculate the diffusion barrier in three
steps:

1. Optimize the initial and final images of minimum energy path (MEP).
2. Generate training data via sampling between the initial and final images.
3. Run NEB calculation to determine the diffusion barrier.

## Step 1: Optimization

Go to the 1-opt directory. There are three python scripts: build.py for building
the un-relaxed initial and final images, and run_opt_is/fs.py to optimize them.

First build the structures:

    ./build.py

Then the optimization:

    ./run_opt_is.py &> is.out &
    ./run_opt_fs.py &> fs.out &

You can also have a look at one of the two files. It simply loads the structure,
instantiates an EMT calculator to the structure and optimize it using
Quasi-Newton algorithm. During the optimization, the initial.traj and final.traj
files produced by build.py will be overwritten.

There are reference files in the 'ref' directory.

## Step 2: Generating training data

In this step we will insert 50 images between the initial and final images and
calculate the energies and forces for all of them. Go to the '2-data' directory
and link the initial/final.traj here from the '1-opt' directory:

    ln -s ../1-opt/initial.traj .
    ln -s ../1-opt/final.traj .

Then run 'run_static.py'

    ./run_static.py &> static.out &

'run_static.py' is simpler than 'run_opt_*.py'. Note that we use the 'IDPP'
interpolation technique to generate the intermediate images and linear
interpolation is likely to yield unphysical structures.


For this example it is very fast. However, it may take hours for common
production-purpose calculation. The python script reports the progress to
standard output, so be patient.

After the calculation, 'static.traj' will be produced, which contains all the
energies and forces. See 'ref' directory for examples of the output.

## Step 3: Running NEB calculation

Finally we come to the 3rd step. Go to the '3-neb' directory and link/copy the
necessary files:

    ln -s ../1-opt/initial.traj .
    ln -s ../1-opt/final.traj .
    cp ../2-data/static.traj train.traj

Note that 'staric.traj' should be renamed to 'train.traj'.

Then we have a look at 'run_neb.py', which seems complicated at the first glance.
This script contains a 'gen_calc_ref' function which is the same as the
'gen_calc' function in 'run_opt_*.py' and 'run_static.py', a 'gen_calc_amp'
function that returns an Amp calculator and a 'gen_args' function that returns
controlling arguments. The reason why we pass function objects instead of
dictionaries/lists is that adaptive controlling arguments are easier to achieve
in this way. Run this script by

    ./run_neb.py &> neb.out &

The progress is reported to the stand output. After a few iterations convergence
will be reached. The MEP is saved in mep.traj, which can be visualized with ASE
GUI

    ase gui mep.traj

Select 'Tools'-'NEB' in the menu, a window will pop out and the diffusion
barrier is shown to be about 0.37 eV, which agrees well with direct NEB result.
