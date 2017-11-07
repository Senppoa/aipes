"""
Functions for determining the optimal topology of artificial neural network, in
order to avoid overfitting.

Available functions
-------------------
validate _energy:
    Calculate RMSE and MaxResid for Amp energies against reference data.
validate_forces:
    Calculate RMSE and MaxResid for Amp forces against reference data.
benchmark:
    Benchmark the model parameters via cross-validation.

CAUTION
-------
When evaluating energies and forces using Amp calculator for comparision with
reference data (typically from first principles calculations), the constraints
should always be turned off by setting apply_constraint=False.

ASE implements constrains in the 'Atoms' class, not in the 'Calculator' class.
So 'apply_constraint=False' is not required when using the calculator to
evaluate energies and forces directly.

If forces are to be included in the training data, be sure to use force-
consistent energies instead of the energies extrapolated to 0 Kelvin. However,
not all the calculators return force-consistent energies directly. In that case,
use the derived calculators in aipes/calculators package with overridden
'get_potential_energy()' method.
"""

import numpy as np

from .utilities import calc_mod, calc_rmse, calc_maxresid, echo


def validate_energy(calc_amp, ref_data):
    """
    Calculate RMSE and MaxResid for Amp energies against reference data.

    Parameters
    ----------
    calc_amp: instance of the 'Amp' class
        Amp calculator to be validated.
    ref_data:
        Reference data to validate the Amp calculator, typically from first
        principles calculations.

    Returns
    -------
    energy_rmse: float
    energy_maxresid: float
        RMSE and maximum residual between the energies predicted by Amp
        calculator and reference data, divided by the number of atoms.
    """
    delta_energy = []
    for image in ref_data:
        energy_ref = image.get_potential_energy(apply_constraint=False)
        energy_amp = calc_amp.get_potential_energy(image)
        delta_energy.append(energy_ref - energy_amp)
    delta_energy = np.array(delta_energy)
    num_atom = len(ref_data[0])
    energy_rmse = calc_rmse(delta_energy) / num_atom
    energy_maxresid = calc_maxresid(delta_energy) / num_atom
    return energy_rmse, energy_maxresid


def validate_forces(calc_amp, ref_data):
    """
    Calculate RMSE and MaxResid for Amp forces against reference data.

    Parameters
    ----------
    calc_amp: 'Amp' object
        Amp calculator to be validated.
    ref_data:
        Reference data to validate the Amp calculator, typically from first
        principles calculations.

    Returns
    -------
    force_rmse: float
    force_maxresid: float
        RMSE and maximum residual between the forces predicted by Amp
        calculator and reference data, divided by (3 * the number of atoms).
    """
    delta_forces_mod = []
    force_maxresid = 0.0
    for image in ref_data:
        forces_ref = image.get_forces(apply_constraint=False)
        forces_amp = calc_amp.get_forces(image)
        delta_forces = forces_ref - forces_amp
        delta_forces_mod.extend([calc_mod(v) for v in delta_forces])
        force_maxresid = np.fmax(force_maxresid, np.max(np.fabs(delta_forces)))
    delta_forces_mod = np.array(delta_forces_mod)
    force_rmse = calc_rmse(delta_forces_mod) / np.sqrt(3)
    return force_rmse, force_maxresid


def benchmark(gen_calc_amp, hidden_layers, dataset):
    """
    Benchmark the model parameters via cross-validation.

    Parameters
    ----------
    gen_calc_amp: function object
        Function that instantiates an 'Amp' object.
    hidden_layers: list or tuple
        Topology of the artificial neural network.
    dataset: 'Dataset' object
        Grouped dataset for cross-validation.

    Returns
    -------
    None

    Notes
    -----
    Dataset should be grouped before calling this function.
    """
    # Run cross validation to determine the average RMSE and MaxResid
    # for energies and forces for training and validation dataset
    accuracy_train = []
    accuracy_valid = []
    for igroup in range(dataset.ngroup):
        # Instantiate an Amp calculator
        calc_amp = gen_calc_amp(hidden_layers)

        # Train the calculator
        train_set, valid_set = dataset.select(igroup)
        calc_amp.train(images=train_set, overwrite=True)

        # Validate the calculator
        energy_rmse, energy_maxresid = validate_energy(calc_amp, train_set)
        force_rmse, force_maxresid = validate_forces(calc_amp, train_set)
        accuracy_train.append([energy_rmse, energy_maxresid, force_rmse,
                               force_maxresid])

        energy_rmse, energy_maxresid = validate_energy(calc_amp, valid_set)
        force_rmse, force_maxresid = validate_forces(calc_amp, valid_set)
        accuracy_valid.append([energy_rmse, energy_maxresid, force_rmse,
                               force_maxresid])

    # echo
    accuracy_train = np.array(accuracy_train)
    accuracy_valid = np.array(accuracy_valid)
    mean_accuracy_train = np.mean(accuracy_train, axis=0)
    mean_accuracy_valid = np.mean(accuracy_valid, axis=0)
    echo(hidden_layers, end="")
    for data in mean_accuracy_train:
        echo("%13.4e" % data, end="")
    for data in mean_accuracy_valid:
        echo("%13.4e" % data, end="")
    echo()
