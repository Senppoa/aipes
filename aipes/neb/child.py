"""
Script to start child process for NEB calculation, called by the 'dynamic'
module.
"""

from mpi4py import MPI
from ase.neb import NEB
from ase.optimize import BFGS, FIRE
from amp import Amp


# Initialize MPI environment
comm_parent = MPI.Comm.Get_parent()
comm_world = MPI.COMM_WORLD
rank_parent = comm_parent.Get_rank()
rank_world = comm_world.Get_rank()
assert rank_parent == rank_world

# Fetch data from parent process
neb_args = comm_parent.bcast(None, root=0)
mep = comm_parent.bcast(None, root=0)
label = comm_parent.bcast(None, root=0)

# Prepare MEP for NEB calculation
calc_amp = Amp.load(label+".amp", cores=1, label=label, logging=False)
mep[rank_world+1].set_calculator(calc_amp)

# Run NEB
neb_runner = NEB(mep,
                 k=neb_args["k"],
                 climb=neb_args["climb"],
                 remove_rotation_and_translation=
                 neb_args["remove_rotation_and_translation"],
                 method=neb_args["method"],
                 parallel=True)
neb_runner.interpolate(neb_args["interp"])
if neb_args["opt_algorithm"] == "FIRE":
    opt_algorithm = FIRE
else:
    opt_algorithm = BFGS
opt_runner = opt_algorithm(neb_runner)
opt_runner.run(fmax=neb_args["fmax"], steps=neb_args["steps"])

# Send MEP back to parent process
active_image = mep[rank_world+1].copy()
comm_parent.gather(active_image, root=0)
comm_parent.Disconnect()