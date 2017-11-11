"""
Shared subroutines by various versions of run_aineb.

Available modules
-----------------
initialize_mep:
    Build the initial minimum energy path (MEP) from initial and final images.
validate_mep:
    Check the difference between energies/forces produced by Amp and first
    principles calculators for images along the MEP to determine if convergence
    has been reached.
"""

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

    # We assume that mep DOES NOT contain initial and final images, which is the
    # case for parallel version of run_aineb. For serial version, pass mep[1:-1]
    # instead of the whole mep to this function as the argument.
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
