#! /usr/bin/env python

import numpy as np
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.io import write


def run_opt(atoms):
    qn = QuasiNewton(atoms)
    qn.run(fmax=0.05)


# Build and relax the inital state
initial = fcc100('Al', size=(2,2,3))
add_adsorbate(initial, 'Au', 1.7, 'hollow')
initial.center(axis=2, vacuum=4.0)

mask = [atom.tag > 1 for atom in initial]
initial.set_constraint(FixAtoms(mask=mask))

initial.set_calculator(EMT())
run_opt(initial)
write("initial.traj", initial)

# Build and relax the final stat
final = initial.copy()
final[-1].x += initial.get_cell()[0,0] / 2

final.set_calculator(EMT())
run_opt(final)
write("final.traj", final)

# Interpolate
ninter = 50
displacement = final.get_positions() - initial.get_positions()
images = []
for ita in np.linspace(0.0, 1.0, ninter+2):
    image_copy = initial.copy()
    image_copy.positions += ita * displacement
    images.append(image_copy)

# Save
write("images.traj", images)
