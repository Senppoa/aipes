"""
AIPES is a library that combines the subroutines provided by the atomic
simulation environment (ASE), such as the dimer method, nudged elastic band
(NEB), molecular dynamics (MD), and global optimization (GO), with machine
learning techniques. The key idea is to predict the potential energy surface
(PES) with machine learning based calculator trained from reference data
generated from first principles calculations. The predicted PES are then
validated using first-principles based reference calculator, and new train data
will be added to the training dataset to gradually improve the prediction
quality.

Currently only NEB has been implemented, and the only supported machine learning
calculator is Amp. There's still much work to do.

Homepage of ASE: https://wiki.fysik.dtu.dk/ase/
Homepage of Amp: https://bitbucket.org/andrewpeterson/amp


Available subpackages
---------------------
calculators:
    Classes derived from ase.calculators.*, with overridden
    'get_potential_energy' method that always return force-consistent
    potential energy regardless of the 'force_consistent' argument.
common:
    Shared subroutines.
neb:
    Subroutines for NEB calculation. Both serial and parallel versions provided.
"""

__version__ = "0.0.1"
