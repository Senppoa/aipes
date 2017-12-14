#! /usr/bin/env python
"""Build the initial and final images."""

from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.io import write


# Build the initial image
initial = fcc100('Al', size=(2, 2, 3))
add_adsorbate(initial, 'Au', 1.7, 'hollow')
initial.center(axis=2, vacuum=4.0)
mask = [atom.tag > 1 for atom in initial]
initial.set_constraint(FixAtoms(mask=mask))
write("initial.traj", initial)

# Build the final image
final = initial.copy()
final[-1].x += initial.get_cell()[0, 0] / 2
write("final.traj", final)
