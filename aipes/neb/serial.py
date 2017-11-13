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
    Amp calculators cannot be shared by more than one NEB images. So we have to
    train it and then load it from disk for each of the images. In this case,
    Amp calculators must be trained with 'overwrite=True' argument.
    """
    # Load the initial and final images and training dataset
    initial_image = read(initial_file, index=-1)
    final_image = read(final_file, index=-1)
    train_set = read(train_file, index=":")

    # Main loop
    echo("Serial AI-NEB running on 1 process.")
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
            mep = read("mep.traj", index=":")
        for image in mep[1:-1]:
            calc_amp = Amp.load(label + ".amp", cores=1, label=label,
                                logging=False)
            image.set_calculator(calc_amp)

        # Calculate the MEP from initial guess
        echo("Running NEB using the Amp calculator...")
        neb_runner = NEB(mep,
                         k=neb_args["k"],
                         climb=neb_args["climb"],
                         remove_rotation_and_translation=
                         neb_args["remove_rotation_and_translation"],
                         method=neb_args["method"])
        if ((iteration == 0 and neb_args["restart"] is False) or
           (iteration != 0 and neb_args["reuse_mep"] is False)):
            neb_runner.interpolate(neb_args["interp"])
        if neb_args["opt_algorithm"] == "FIRE":
            opt_algorithm = FIRE
        else:
            opt_algorithm = BFGS
        opt_runner = opt_algorithm(neb_runner)
        opt_runner.run(fmax=neb_args["fmax"], steps=neb_args["steps"])

        # Validate the MEP against the reference calculator
        # Note that for serial version of run_aineb we have to pass mep[1:-1]
        # to validate_mep instead of the whole mep.
        echo("Validating the MEP using reference calculator...")
        accuracy, ref_images = validate_mep(mep[1:-1], calc_amp, gen_calc_ref)
        converge_status = []
        for key, value in accuracy.items():
            echo("%16s = %13.4e" % (key, value))
            converge_status.append(value <= convergence[key])

        # Save the MEP
        # Note that this piece of code MUST be placed here. Otherwise the
        # energies in mep.traj will be lost.
        mep_save = [initial_image]
        mep_save.extend(ref_images)
        mep_save.append(final_image)
        write("mep.traj", mep_save)

        # Check if convergence has been reached
        if converge_status == [True, True, True, True]:
            is_converged = True
            break
        else:
            train_set.extend(ref_images)
            write("train_new.traj", train_set)

    # Summary
    if is_converged:
        echo("\nAI-NEB calculation converged."
             "\nThe MEP is saved in mep.traj.")
    else:
        echo("\nMaximum iteration number reached."
             "\nAI-NEB calculation not converged.")
