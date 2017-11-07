#! /usr/bin/env python
"""
This program generates fingerprints and their derivatives for the training
data set to boost AI-NEB calculation.
"""

from ase.io import read

from amp.descriptor.cutoffs import Cosine
# from amp.descriptor.zernike import Zernike
from amp.descriptor.gaussian import Gaussian
from amp.regression import Regressor
from amp.model import LossFunction
from amp.model.neuralnetwork import NeuralNetwork
from amp import Amp


def main():
    # Declare controlling parameters
    train_file = "train.traj"

    # Instantiate and train the calculator
    train_set = read(train_file, index=":")
    calc_amp = gen_calc_amp()
    calc_amp.train(images=train_set)


def gen_calc_amp():
    """Returns an Amp calculator."""
    # --------------------------------------------------------------------------
    #
    # Declare controlling parameters
    cutoff_radius = 6.5
    # nmax = 5
    # gs = {"Al": {"Al": 6.0, "Au": 4.0}, "Au": {"Al": 4.0, "Au": 2.0}}
    hidden_layers = (5, 5)
    activation = "tanh"
    optimizer = "BFGS"
    convergence = {"energy_rmse": 0.1,
                   "energy_maxresid": 0.2,
                   "force_rmse": 0.5,
                   "force_maxresid": 1.0}
    checkpoints = 500
    label = "amp/train"
    cores = 10
    logging = True

    # --------------------------------------------------------------------------
    #
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


if __name__ == "__main__":
    main()
