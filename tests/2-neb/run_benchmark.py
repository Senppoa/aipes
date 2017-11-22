#! /usr/bin/env python
"""
This program determines the optimal model parameters of Amp calculator.

Currently only the topology (hidden layers) is considered as the key
convergence parameter. Others may be added later.
"""

from amp.descriptor.cutoffs import Cosine
from amp.descriptor.gaussian import Gaussian
from amp.regression import Regressor
from amp.model import LossFunction
from amp.model.neuralnetwork import NeuralNetwork
from amp import Amp

from aipes.common.dataset import Dataset
from aipes.common.benchmark import benchmark


def main():
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    traj = "train.traj"
    num_group = 5
    num_node_min = 5
    num_node_max = 5
    step = 1

    # --------------------------------------------------------------------------
    # Load and group the data set
    all_data = Dataset(traj)
    all_data.group(num_group, rand=True)

    # Run cross-validation over num_node
    for num_node in range(num_node_min, num_node_max+step, step):
        hidden_layers = (num_node, num_node)
        benchmark(gen_calc_amp, hidden_layers, all_data)


def gen_calc_amp(hidden_layers):
    """Returns an Amp calculator."""
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    cutoff_radius = 6.5
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
