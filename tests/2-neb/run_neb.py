#! /usr/bin/env python
"""Main part of the AINEB program."""

from ase.calculators.emt import EMT

from amp.descriptor.cutoffs import Cosine
# from amp.descriptor.zernike import Zernike
from amp.descriptor.gaussian import Gaussian
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
    num_transition_states = 5

    train_file = "train.traj"

    convergence = {"energy_rmse": 0.001,
                   "energy_maxresid": 0.002,
                   "force_rmse": 0.05,
                   "force_maxresid": 0.1,
                   "max_iteration": 1000}

    neb_args = {"climb": False,
                "method": "aseneb",
                "interp": "linear",
                "fmax": 0.05,
                "steps": 50,
                "reuse_mep": True}

    # --------------------------------------------------------------------------
    # Run the job
    run_aineb(initial_file, final_file, num_transition_states,
              train_file, convergence, neb_args,
              gen_calc_amp, gen_calc_ref)
    

def gen_calc_amp():
    """Returns an Amp calculator."""
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    cutoff_radius = 6.5
    # nmax = 5
    # gs = {"Al": {"Al": 6.0, "Au": 4.0}, "Au": {"Al": 4.0, "Au": 2.0}}
    hidden_layers = (5, 5)
    activation = "tanh"
    optimizer = "BFGS"
    convergence = {"energy_rmse": 0.001,
                   "energy_maxresid": 0.002,
                   "force_rmse": 0.05,
                   "force_maxresid": 0.1}
    checkpoints = 500
    label = "amp/train"
    cores = 2
    logging = True

    # --------------------------------------------------------------------------
    # Instantiate the descriptor
    cutoff = Cosine(cutoff_radius)
    # descriptor = Zernike(cutoff=cutoff, Gs=gs, nmax=nmax)
    descriptor = Gaussian(cutoff=cutoff)

    # Instantiate the model
    regressor = Regressor(optimizer=optimizer)
    lossfunction = LossFunction(convergence=convergence)
    model = NeuralNetwork(hiddenlayers=hidden_layers, activation=activation,
                          regressor=regressor, lossfunction=lossfunction,
                          checkpoints=checkpoints, mode="atom-centered")

    # Instantiate the Amp calculator
    calc = Amp(descriptor=descriptor, model=model, label=label, cores=cores,
               logging=logging)

    return calc


def gen_calc_ref():
    """Return a reference calculator."""
    calc = EMT()
    return calc


if __name__ == "__main__":
    main()
