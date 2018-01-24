"""
Serial version of NEB method based on machine learning PES with first
principles correction.

Available functions
-------------------
run_aineb:
    Performs NEB calculation with first principles corrections.
"""

from ase.io import read, write
from ase.neb import NEB
from ase.optimize import BFGS, FIRE

from amp import Amp
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
    Amp calculators cannot be shared by more than one NEB images. So we have to
    train it and then load it from disk for each of the images. In this case,
    Amp calculators must be trained with 'overwrite=True' argument.
    """
    # Generate the initial controlling arguments
    mep_args, control_args, dataset_args, convergence, neb_args = gen_args()

    # Load the initial and final images and training dataset
    initial_image = read(mep_args["initial_file"], index=-1, parallel=False)
    final_image = read(mep_args["final_file"], index=-1, parallel=False)
    full_set = read(dataset_args["train_file"], index=":", parallel=False)
    train_set = cluster_data(full_set, dataset_args)

    # Main loop
    echo("Serial AI-NEB running on 1 process.")
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
            mep = read("mep.traj", index=":")
        for image in mep[1:-1]:
            calc_amp = Amp.load(label + ".amp", cores=1, label=label,
                                logging=False)
            image.set_calculator(calc_amp)

        # Calculate the MEP from initial guess
        echo("Running NEB using the Amp calculator...")
        assert (len(neb_args["climb"]) ==
                len(neb_args["opt_algorithm"]) ==
                len(neb_args["fmax"]) ==
                len(neb_args["steps"]))
        for stage in range(len(neb_args["climb"])):
            if neb_args["climb"][stage] is False:
                echo("Climbing image switched off.")
            else:
                echo("Climbing image switched on.")
            neb_runner = NEB(mep,
                             k=neb_args["k"],
                             climb=neb_args["climb"][stage],
                             remove_rotation_and_translation=
                             neb_args["rm_rot_trans"],
                             method=neb_args["method"])
            # NOTE: interpolation is done in initialize_mep.
            if neb_args["opt_algorithm"][stage] == "FIRE":
                opt_algorithm = FIRE
            else:
                opt_algorithm = BFGS
            opt_runner = opt_algorithm(neb_runner)
            opt_runner.run(fmax=neb_args["fmax"][stage],
                           steps=neb_args["steps"][stage])

        # The validation of MEP is very time-consuming. Here we save MEP without
        # energies and forces for inspection.
        write("chk.traj", mep, parallel=False)

        # Validate the MEP against the reference calculator
        # Note that for serial version of run_aineb we have to pass mep[1:-1]
        # to validate_mep instead of the whole mep.
        echo("Validating the MEP using reference calculator...")
        accuracy, ref_images = validate_mep(mep[1:-1], calc_amp, gen_calc_ref)
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
