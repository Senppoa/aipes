#! /usr/bin/env python
"""Main part of the AINEB program."""

from ase.calculators.emt import EMT

from amp.descriptor.cutoffs import Cosine
from amp.descriptor.gaussian import Gaussian
# from amp.descriptor.zernike import Zernike
from amp.regression import Regressor
from amp.model import LossFunction
from amp.model.neuralnetwork import NeuralNetwork
from amp import Amp

from aipes.neb.dynamic import run_aineb


def main():
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    initial_file = "initial.traj"
    final_file = "final.traj"
    num_inter_images = 5

    # --------------------------------------------------------------------------
    # Run the job
    run_aineb(initial_file, final_file, num_inter_images,
              gen_args, gen_calc_amp, gen_calc_ref)


def gen_args(iteration=0, accuracy=None):
    """
    Generate controlling arguments adaptively.

    Parameters
    ----------
    iteration: integer
        Iteration number.
    accuracy: dictionary
        Discrepancy between the energies and forces as predicted by Amp and ab
        initio calculators for images on the MEP.

    Returns
    -------
    control_args: dictionary
        Arguments controlling the restart and reuse behaviors.
    dataset_args: dictionary
        Arguments controlling the training dataset.
    convergence: dictionary
        Convergence criteria.
    neb_args: dictionary
        Arguments controlling the NEB calculation.
    """
    # Default settings arguments
    control_args = {
        "restart_with_calc": False,
        "restart_with_mep": False,
        "reuse_calc": True,
        "reuse_mep": False
    }

    dataset_args = {
        "train_file": "train.traj",
        "image_fmax": 10.0
    }

    convergence = {
        "energy_rmse": 0.001,
        "energy_maxresid": 0.002,
        "force_rmse": 0.05,
        "force_maxresid": 0.10,
        "max_iteration": 1000
    }

    neb_args = {
        "k": 5.0,
        "method": "improvedtangent",
        "interp": "idpp",
        "mic": True,
        "rm_rot_trans": False,
        "climb": [False, True],
        "opt_algorithm": ["BFGS", "FIRE"],
        "fmax": [0.5, 0.05],
        "steps": [10, 40],
    }

    # Adjust arguments according to iteration number and accuracy
    # if iteration > 0 and accuracy["force_maxresid"] <= 0.5:
    #     neb_args["steps"] = [10, 100]

    return control_args, dataset_args, convergence, neb_args


def gen_calc_amp(reload=False):
    """Returns an Amp calculator."""
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    cutoff_radius = 6.5
    hidden_layers = (5, 5)
    activation = "tanh"
    optimizer = "BFGS"
    convergence = {
        "energy_rmse": 0.001,
        "energy_maxresid": 0.002,
        "force_rmse": 0.05,
        "force_maxresid": 0.10
    }
    checkpoints = 500
    label = "amp/train"
    cores = 20
    logging = True

    # --------------------------------------------------------------------------
    # Instantiate the descriptor
    cutoff = Cosine(cutoff_radius)
    descriptor = Gaussian(cutoff=cutoff)
    # descriptor = Zernike(cutoff=cutoff)

    # Instantiate the model
    regressor = Regressor(optimizer=optimizer)
    lossfunction = LossFunction(convergence=convergence)
    model = NeuralNetwork(hiddenlayers=hidden_layers, activation=activation,
                          regressor=regressor, lossfunction=lossfunction,
                          checkpoints=checkpoints, mode="atom-centered")

    # Instantiate the Amp calculator
    if reload is False:
        calc = Amp(descriptor=descriptor, model=model, label=label, cores=cores,
                   logging=logging)
    else:
        calc = Amp.load(label+".amp", cores=cores, label=label, logging=logging)
        calc.model.regressor = regressor
        calc.model.lossfunction = lossfunction
    return calc


def gen_calc_ref():
    """Return a reference calculator."""
    calc = EMT()
    return calc


if __name__ == "__main__":
    main()
