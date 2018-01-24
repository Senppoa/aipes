"""
Shared subroutines by various versions of run_aineb.

Available functions
-------------------
initialize_mep:
    Build the initial minimum energy path (MEP) from initial and final images.
validate_mep:
    Check the difference between energies/forces produced by Amp and first
    principles calculators for images along the MEP to determine if convergence
    has been reached.
cluster_data:
    Cluster the full data into training and remaining datasets in order to avoid
    fitting problems.
"""

import time

import numpy as np

from ase.neb import NEB

from ..common.utilities import echo
from ..common.benchmark import validate_energy, validate_forces


def initialize_mep(initial_image, final_image, num_inter_images, neb_args):
    """Build the MEP from initial and final images."""
    mep = [initial_image]
    for i in range(num_inter_images):
        mep.append(initial_image.copy())
    mep.append(final_image)
    neb_runner = NEB(mep)
    neb_runner.interpolate(method=neb_args["interp"], mic=neb_args["mic"])
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
    for index, image in enumerate(mep):
        t0 = time.strftime("%H:%M:%S")
        echo("Dealing with image # %d at %s." % (index+1, t0))
        image_copy = image.copy()
        image_copy.set_calculator(gen_calc_ref())

        # For the first few iterations NEB may produce unphysical intermediate
        # images, and the forces and energy calls are likely to fail. We have to
        # handle this exception here.
        try:
            image_copy.get_forces(apply_constraint=False)
            image_copy.get_potential_energy(apply_constraint=False)
        except RuntimeError:
            echo("ERROR: reference code exited abnormally.")
            echo("Image discarded.")
            pass
        except UnboundLocalError:
            echo("ERROR: forces/energy evaluation failed.")
            echo("Image discarded.")
            pass
        else:
            ref_images.append(image_copy)

    # Calculate RMSE and MaxResid
    energy_rmse, energy_maxresid = validate_energy(calc_amp, ref_images)
    force_rmse, force_maxresid = validate_forces(calc_amp, ref_images)
    accuracy = {"energy_rmse": energy_rmse, "energy_maxresid": energy_maxresid,
                "force_rmse": force_rmse, "force_maxresid": force_maxresid}

    return accuracy, ref_images


def cluster_data(full_set, dataset_args):
    """
    Cluster the full data into training and remaining datasets in order to avoid
    fitting problems. Only the training dataset is returned.
    """
    mean_epot = np.array([image.get_potential_energy(apply_constraint=False)
                          for image in full_set]).mean()
    train_set = []
    for image in full_set:
        forces = image.get_forces(apply_constraint=False)
        forces_mod = np.array([np.sqrt(np.sum(v**2)) for v in forces])
        epot = image.get_potential_energy(apply_constraint=False)
        if (np.max(forces_mod) <= dataset_args["image_fmax"] and
           abs(epot-mean_epot) <= dataset_args["image_dE"]):
            train_set.append(image)
    return train_set
