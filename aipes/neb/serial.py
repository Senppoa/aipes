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

from amp import Amp

from ..common.utilities import echo
from ..common.benchmark import validate_energy, validate_forces


def initialize_mep(initial_image, final_image, num_inter_images):
    """Build the MEP from initial and final images."""
    mep = [initial_image]
    for i in range(num_inter_images):
        mep.append(initial_image.copy())
    mep.append(final_image)
    return mep


def validate_mep(mep, calc_amp, gen_calc_ref):
    """
    Check MEP against reference calculator.

    Parameters
    ----------
    mep: list of 'atoms' objects
        MEP produced by NEB calculation with Amp calculator. To be checked for
        convergence.
    calc_amp: 'Amp' object
        Amp calculator with which NEB calculation has been performed.
    gen_calc_ref: function object
        Function than instantiates a reference (first principles) calculator.

    Returns
    -------
    accuracy: dictionary
        Energy_rmse, energy_maxresid, force_rmse, force_maxresid determined from
        the difference between energies/forces produced by Amp and first
        principles calculations.
    ref_images: list of 'Atoms' objects
        Accurate energies and forces along MEP to improve the training data set.

    CAUTION
    -------
    We have to assign a separate reference calculator to each copy of the images
    in MEP. This is due to the fact that ASE stores forces and energies in the
    calculator, not in atoms. Sharing the same calculator among multiple images
    will cause re-calculation and wastes time.

    We shall not apply any constraints when comparing energies and forces from
    Amp calculator against the results from reference calculator.
    """
    ref_images = []

    # For parallel calculation the MEP passed in does not contain initial or
    # final states, different from that in serial calculation.
    for image in mep:
        image_copy = image.copy()
        image_copy.set_calculator(gen_calc_ref())

        # The energy may be extracted at the same time as evaluating forces.
        # So we calculate the forces first to save time.
        image_copy.get_forces(apply_constraint=False)
        image_copy.get_potential_energy(apply_constraint=False)

        # Append image_copy to ref_images for improving the training database
        ref_images.append(image_copy)

    # Calculate RMSE and MaxResid
    energy_rmse, energy_maxresid = validate_energy(calc_amp, ref_images)
    force_rmse, force_maxresid = validate_forces(calc_amp, ref_images)
    accuracy = {"energy_rmse": energy_rmse, "energy_maxresid": energy_maxresid,
                "force_rmse": force_rmse, "force_maxresid": force_maxresid}

    return accuracy, ref_images


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
    is_converged = False
    for iteration in range(convergence["max_iteration"]):
        echo("\nIteration # %d" % (iteration+1))

        # Train the Amp calculator
        echo("Training the Amp calculator...")
        calc_amp = gen_calc_amp()
        calc_amp.train(images=train_set, overwrite=True)
        label = calc_amp.label

        # Build the initial MEP
        mep = initialize_mep(initial_image, final_image, num_inter_images)
        for image in mep[1:-1]:
            calc_amp = Amp.load(label + ".amp", cores=1, label=label,
                                logging=False)
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
