"""
Serial version of NEB method based on machine learning PES with first
principles correction.

Available functions
-------------------
initialize_mep:
    Build the initial minimum energy path (MEP) from initial and final images.
validate_mep:
    Check the difference between energies/forces produced by Amp and first
    principles calculators for images along the MEP to determine if convergence
    has been reached.
run_aineb:
    Performs NEB calculation with first principles corrections.
"""

from ase.io import read, write
from ase.neb import NEB
from ase.optimize import BFGS as OPT

from ..common.utilities import echo
from .parallel import initialize_mep, validate_mep


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
    ------
    None
    """
    # Load the initial and final images and training dataset
    initial_image = read(initial_file, index=-1)
    final_image = read(final_file, index=-1)
    train_set = read(train_file, index=":")

    # Main loop
    is_converged = False
    for iteration in range(convergence["max_iteration"]):
        echo("\nIteration # %d" % (iteration+1))

        # Train the Amp calculator
        echo("Training the Amp calculator...")
        calc_amp = gen_calc_amp()
        calc_amp.train(images=train_set, overwrite=True)

        # Build the initial MEP
        mep = initialize_mep(initial_image, final_image, num_inter_images)
        for image in mep[1:-1]:
            image.set_calculator(calc_amp)

        # Calculate the MEP from initial guess
        echo("Performing NEB calculation using the Amp calculator...")
        neb_runner = NEB(mep, climb=neb_args["climb"],
                         method=neb_args["method"])
        neb_runner.interpolate(neb_args["interp"])
        opt_runner = OPT(neb_runner)
        opt_runner.run(fmax=neb_args["fmax"], steps=neb_args["steps"])

        # Validate the MEP against the reference calculator
        echo("Validating the MEP against the reference calculator...")
        accuracy, ref_images = validate_mep(mep, calc_amp, gen_calc_ref)
        converge_status = []
        for key, value in accuracy.items():
            echo("%16s = %13.4e" % (key, value))
            converge_status.append(value <= convergence[key])

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
        mep_final = [initial_image]
        mep_final.extend(ref_images)
        mep_final.append(final_image)
        write("mep.traj", mep_final)
    else:
        echo("\nMaximum iteration number reached."
             "\nAI-NEB calculation not converged.")
