"""
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
