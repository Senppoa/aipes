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

from amp.utilities import Annealer

from ..common.utilities import echo
from .common import initialize_mep, validate_mep, cluster_data


def run_aineb(gen_args, gen_calc_amp, gen_calc_ref):
    """
    Performs NEB calculation with first principles corrections.

    Parameters
    ----------
    gen_args: function object
        Function that generates controlling arguments adaptively.
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
    # Generate the initial controlling arguments
    mep_args, control_args, dataset_args, convergence, neb_args = gen_args()

    # Load the initial and final images and training dataset
    initial_image = read(mep_args["initial_file"], index=-1, parallel=False)
    final_image = read(mep_args["final_file"], index=-1, parallel=False)
    full_set = read(dataset_args["train_file"], index=":", parallel=False)
    train_set = cluster_data(full_set, dataset_args)

    # Main loop
    echo("Dynamic AI-NEB running on %d MPI processes." %
         mep_args["num_inter_images"])
    is_converged = False
    for iteration in range(convergence["max_iteration"]):
        echo("\nIteration # %d" % (iteration+1))

        # Train the Amp calculator
        if ((iteration == 0 and control_args["restart_with_calc"] is False) or
           (iteration != 0 and control_args["reuse_calc"] is False)):
            echo("Initial Amp calculator built from scratch.")
            reload = False
        else:
            echo("Initial Amp calculator loaded from previous training.")
            reload = True
        calc_amp = gen_calc_amp(reload=reload)
        echo("Training the Amp calculator...")
        if control_args["annealing"] is True:
            Annealer(calc=calc_amp, images=train_set)
        calc_amp.train(images=train_set, overwrite=True)
        label = calc_amp.label

        # Build the initial MEP
        if ((iteration == 0 and control_args["restart_with_mep"] is False) or
           (iteration != 0 and control_args["reuse_mep"] is False)):
            echo("Initial MEP built from scratch.")
            mep = initialize_mep(initial_image, final_image,
                                 mep_args["num_inter_images"], neb_args)
        else:
            echo("Initial MEP loaded from mep.traj.")
            mep = read("mep.traj", index=":", parallel=False)

        # Spawn MPI child processes to calculate the MEP from initial guess
        echo("Running NEB using the Amp calculator...")
        comm = MPI.COMM_SELF.Spawn(sys.executable,
                                   args=["-m", "aipes.neb.child"],
                                   maxprocs=mep_args["num_inter_images"])
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

        # Update training dataset
        full_set.extend(ref_images)
        train_set = cluster_data(full_set, dataset_args)
        echo("Size of training dataset after clustering: %d." % len(train_set))
        write("train_new.traj", full_set, parallel=False)

        # Update controlling arguments
        (mep_args, control_args, dataset_args,
         convergence, neb_args) = gen_args(iteration+1, accuracy)

        # Check if convergence has been reached
        if converge_status == [True, True, True, True]:
            is_converged = True
            break

    # Summary
    if is_converged:
        echo("\nAI-NEB calculation converged."
             "\nThe MEP is saved in mep.traj.")
    else:
        echo("\nMaximum iteration number reached."
             "\nAI-NEB calculation not converged.")
