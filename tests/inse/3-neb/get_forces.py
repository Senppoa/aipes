#! /usr/bin/env python
"""Get fmax for each iamge in training data set."""

import sys

import numpy as np
from ase.io import read


images = read(sys.argv[1], index=':')
for image in images:
    forces = image.get_forces(apply_constraint=False)
    forces_mod = [np.sqrt(np.sum(v**2)) for v in forces]
    forces_mod = np.array(forces_mod)
    print(np.max(forces_mod))
