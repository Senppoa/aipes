"""
Parallel version of NEB method based on machine learning PES with first
principles correction, with dynamic process management.

Available functions
-------------------
run_aineb:
    Performs NEB calculation with first principles corrections.
"""

import sys

from ase.io import read, write

from mpi4py import MPI

from ..common.utilities import echo
from .common import initialize_mep, validate_mep


def run_aineb(initial_file, final_file, num_inter_images,
              train_file, convergence, neb_args,
              gen_calc_amp, gen_calc_ref):
    """
    Performs NEB calculation with first principles corrections.

    Parameters
    ----------
    initial_file, final_file: ASE trajectory files
        Trajectory files specifying the initial and final images of MEP. Should
        have been assigned with single-point calculators which records reference
        energies.
    num_inter_images: integer
        Number of intermediate images between initial and final images of MEP.
    train_file: ASE trajectory file
        Trajectory containing the training data. Generated from first principle
        calculations.
    convergence: dictionary
        Convergence criteria.
    neb_args: dictionary
        Arguments controlling the NEB calculation.
    gen_calc_amp: function object
        Function that instantiates an Amp calculator.
    gen_calc_ref: function object
        Function that instantiates a reference (first principles) calculator.

    Returns
    -------
    None

    CAUTION
    -------
    We have to use parallel=False otherwise the MPI environment will be broken
    and comm.bcast() will always fail. This annoying BUG has cost me many hours.

    Amp calculator cannot be passed via MPI and will produce errors like
    "TypeError: cannot serialize '_io.TextIOWrapper'. So we have to train the
    Amp calculator on parent process, write it to disk and then have all the
    child processes reload it. In this case, Amp calculators must be trained
    with 'overwrite=True' argument.
    """
    # Load the initial and final images and training dataset
    initial_image = read(initial_file, index=-1, parallel=False)
    final_image = read(final_file, index=-1, parallel=False)
    train_set = read(train_file, index=":", parallel=False)

    # Main loop
    echo("Dynamic AI-NEB running on %d MPI processes." % num_inter_images)
    is_converged = False
    for iteration in range(convergence["max_iteration"]):
        echo("\nIteration # %d" % (iteration+1))

        # Train the Amp calculator
        echo("Training Amp calculator...")
        calc_amp = gen_calc_amp()
        calc_amp.train(images=train_set, overwrite=True)
        label = calc_amp.label

        # Build the initial MEP
        if ((iteration == 0 and neb_args["restart"] is False) or
           (iteration != 0 and neb_args["reuse_mep"] is False)):
            echo("Initial MEP built from scratch.")
            mep = initialize_mep(initial_image, final_image, num_inter_images)
        else:
            echo("Initial MEP loaded from mep.traj.")
            mep = read("mep.traj", index=":", parallel=False)

        # Spawn MPI child processes and run NEB
        echo("Running NEB using the Amp calculator...")
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=["-m", "aipes.neb.child"],
                                   maxprocs=num_inter_images)
        comm.bcast(iteration, root=MPI.ROOT)
        comm.bcast(neb_args, root=MPI.ROOT)
        comm.bcast(mep, root=MPI.ROOT)
        comm.bcast(label, root=MPI.ROOT)
        active_image = None
        mep = comm.gather(active_image, root=MPI.ROOT)
        comm.Disconnect()

        # Validate the MEP against the reference calculator
        echo("Validating the MEP using reference calculator...")
        accuracy, ref_images = validate_mep(mep, calc_amp, gen_calc_ref)
        converge_status = []
        for key, value in accuracy.items():
            echo("%16s = %13.4e (%13.4e)" % (key, value, convergence[key]))
            converge_status.append(value <= convergence[key])

        # Save the MEP
        # Note that this piece of code MUST be placed here. Otherwise the
        # energies in mep.traj will be lost.
        mep_save = [initial_image]
        mep_save.extend(ref_images)
        mep_save.append(final_image)
        write("mep.traj", mep_save, parallel=False)

        # Check if convergence has been reached
        if converge_status == [True, True, True, True]:
            is_converged = True
            break
        else:
            train_set.extend(ref_images)
            write("train_new.traj", train_set, parallel=False)

    # Summary
    if is_converged:
        echo("\nAI-NEB calculation converged."
             "\nThe MEP is saved in mep.traj.")
    else:
        echo("\nMaximum iteration number reached."
             "\nAI-NEB calculation not converged.")
