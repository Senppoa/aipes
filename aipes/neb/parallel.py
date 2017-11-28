"""
Parallel version of NEB method based on machine learning PES with first
principles correction.

Available functions
-------------------
run_aineb:
    Performs NEB calculation with first principles corrections.
"""

from ase.io import read, write
from ase.neb import NEB
from ase.optimize import BFGS, FIRE

from mpi4py import MPI

from amp import Amp

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
    Amp calculator on master process, write it to disk and then have all the
    processes reload it. In this case, Amp calculators must be trained with
    'overwrite=True' argument.
    """
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size == num_inter_images

    # Load the initial and final images and training dataset
    if rank == 0:
        initial_image = read(initial_file, index=-1, parallel=False)
        final_image = read(final_file, index=-1, parallel=False)
        train_set = read(train_file, index=":", parallel=False)

    # Main loop
    echo("Parallel AI-NEB running on %d MPI processes." % size, rank)
    is_converged = False
    for iteration in range(convergence["max_iteration"]):
        echo("\nIteration # %d" % (iteration+1), rank)

        # Train the Amp calculator
        # While the master process is training the calculator, we call
        # comm.bcast() to suspend the other processes.
        if rank == 0:
            if ((iteration == 0 and neb_args["restart_with_calc"] is False) or
               (iteration != 0 and neb_args["reuse_calc"] is False)):
                echo("Initial Amp calculator built from scratch.", rank)
                reload = False
            else:
                echo("Initial Amp calculator loaded from previous training.",
                     rank)
                reload = True
            calc_amp = gen_calc_amp(reload=reload)
            echo("Training the Amp calculator...", rank)
            calc_amp.train(images=train_set, overwrite=True)
            label = calc_amp.label
        else:
            label = None
        label = comm.bcast(label, root=0)
        calc_amp = Amp.load(label+".amp", cores=1, label=label, logging=False)

        # Build the initial MEP
        if rank == 0:
            if ((iteration == 0 and neb_args["restart_with_mep"] is False) or
               (iteration != 0 and neb_args["reuse_mep"] is False)):
                echo("Initial MEP built from scratch.", rank)
                mep = initialize_mep(initial_image, final_image,
                                     num_inter_images, neb_args["interp"])
            else:
                echo("Initial MEP loaded from mep.traj.", rank)
                mep = read("mep.traj", index=":", parallel=False)
        else:
            mep = None
        mep = comm.bcast(mep, root=0)
        mep[rank+1].set_calculator(calc_amp)

        # Calculate the MEP from initial guess
        echo("Running NEB using the Amp calculator...", rank)
        assert (len(neb_args["climb"]) ==
                len(neb_args["opt_algorithm"]) ==
                len(neb_args["fmax"]) ==
                len(neb_args["steps"]))
        for stage in range(len(neb_args["climb"])):
            if neb_args["climb"][stage] is False:
                echo("Climbing image switched off.", rank)
            else:
                echo("Climbing image switched on.", rank)
            neb_runner = NEB(mep,
                             k=neb_args["k"],
                             climb=neb_args["climb"][stage],
                             remove_rotation_and_translation=
                             neb_args["rm_rot_trans"],
                             method=neb_args["method"],
                             parallel=True)
            # NOTE: interpolation is done in initialize_mep.
            if neb_args["opt_algorithm"][stage] == "FIRE":
                opt_algorithm = FIRE
            else:
                opt_algorithm = BFGS
            opt_runner = opt_algorithm(neb_runner)
            opt_runner.run(fmax=neb_args["fmax"][stage],
                           steps=neb_args["steps"][stage])
        # Amp calculator cannot be passed by MPI, so we have to gather a copy of
        # the image.
        mep = comm.gather(mep[rank+1].copy(), root=0)

        # Validate the MEP against the reference calculator
        echo("Validating the MEP using reference calculator...", rank)
        if rank == 0:
            accuracy, ref_images = validate_mep(mep, calc_amp, gen_calc_ref)
            converge_status = []
            for key, value in accuracy.items():
                echo("%16s = %13.4e (%13.4e)" % (key, value, convergence[key]),
                     rank)
                converge_status.append(value <= convergence[key])
        else:
            converge_status = None
        converge_status = comm.bcast(converge_status, root=0)

        # Save the MEP
        # Note that this piece of code MUST be placed here. Otherwise the
        # energies in mep.traj will be lost.
        if rank == 0:
            mep_save = [initial_image]
            mep_save.extend(ref_images)
            mep_save.append(final_image)
            write("mep.traj", mep_save, parallel=False)

        # Check if convergence has been reached
        if converge_status == [True, True, True, True]:
            is_converged = True
            break
        else:
            if rank == 0:
                train_set.extend(ref_images)
                write("train_new.traj", train_set, parallel=False)

    # Summary
    if is_converged:
        echo("\nAI-NEB calculation converged."
             "\nThe MEP is saved in mep.traj.", rank)
    else:
        echo("\nMaximum iteration number reached."
             "\nAI-NEB calculation not converged.", rank)
