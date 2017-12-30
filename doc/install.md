# Installation

## Dependencies

AIPES is compatible with python 3.x (x>4) and requires the following libraries
* Atomic Simulation Environment (https://wiki.fysik.dtu.dk/ase).
* Atomic Machine-learning package (https://bitbucket.org/andrewpeterson/amp).
* MPI4PY (https://pypi.python.org/pypi/mpi4py/3.0.0)

Each library has its own dependencies, and the installation guides can be found
on their websites. For convenience, it is recommended to install Anaconda from
https://www.anaconda.com, which bundles Numpy, SciPy, matplotlib, nose, Pexpect
and PyZMQ in a single installer.

When installing the Amp package, it is recommend to install Pexpect and PyZMQ in
order to enable parallel training.

After the installation of ASE, add the following lines to the head of 
parallel.py under the installation directory:

    10 from ase.utils import devnull
    11 
    12 try:
    13     import mpi4py
    14 except ImportError:
    15     pass
    16 
    17 
    18 def get_txt(txt, rank):

Here line 12-15 are newly added lines. This is due to the fact that ASE supports
many parallelism libraries while AIPES only supports MPI4PY. So we have to
short-circuit some subroutines in parallel.py.


## Set the environment

After the successful installation and test of the dependencies, simply unpack
the source code and add the top 'aipes' directory to the *PYTHONPATH*
environment variable. For example, if the top directory of unpacked source code
is $HOME/soft/aipes, and there are 'aipes', 'doc', 'tests', 'LICENSE' and
'README.md' under this directory, then add the following settings to
$HOME/.bashrc:

    export PYTHONPATH=$HOME/soft/aipes:$PYTHONPATH

## Test the installation

There are two examples under the 'tests' directory.Au@Al001 calculates the
diffusion barrier on Al001 surface. This test uses the empirical EMT calculator
and takes about half an hour to finish if executed in parallel.
The other example H2O@GaSe calculates the disassociation barrier of a water
molecule at a Se vacancy on two-dimensional GaSe surface. This test requires the
installation of VASP and takes about 30 hours to finish on 16 Intel(R) Xeon(R)
E5-2680 v2 CPUs if executed in parallel. See tutorials.md for more detailed
instructions.