"""
Derived VASP calculator class with overridden get_potential_energy() method.
"""

from ase.calculators.vasp import Vasp


class VaspFC(Vasp):
    def get_potential_energy(self, atoms, force_consistent=False):
        self.update(atoms)
        return self.energy_free
