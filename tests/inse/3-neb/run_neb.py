#! /usr/bin/env python
"""Main part of the AINEB program."""

import os

from aipes.calculators.vasp import VaspFC as Vasp

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
    num_inter_images = 4

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
        "force_maxresid": 0.15,
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
        "fmax": [0.5, 0.1],
        "steps": [10, 40],
    }

    # --------------------------------------------------------------------------
    # Run the job
    run_aineb(initial_file, final_file, num_inter_images,
              control_args, dataset_args, convergence, neb_args,
              gen_calc_amp, gen_calc_ref)
    

def gen_calc_amp(reload=False):
    """Returns an Amp calculator."""
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    cutoff_radius = 6.5
    hidden_layers = (10, 10)
    activation = "tanh"
    optimizer = "BFGS"
    convergence = {
        "energy_rmse": 0.001,
        "energy_maxresid": 0.002,
        "force_rmse": 0.05,
        "force_maxresid": 0.15
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
    os.environ["VASP_PP_PATH"] = "/home/yhli/soft/pp.v54"
    os.environ["VASP_COMMAND"] = "mpirun -np 20 vasp_std &> vasp.log"
    calc = Vasp(restart=False, xc="PBE", kpts=[3, 3, 1], gamma=True,
                prec="Accurate", 
                ispin=2, lvdw=True,
                encut=400, nelm=60, ediff=1.e-4, nelmin=6, lreal="Auto",
                nsw=0, isym=0, ismear=0, sigma=0.05)
    return calc


if __name__ == "__main__":
    main()
