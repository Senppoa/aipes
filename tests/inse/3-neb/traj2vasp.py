#! /usr/bin/env python
"""Convert mep.traj to POSCAR.#N."""

from ase.io import read, write


mep = read("mep.traj", index=":")
for i, image in enumerate(mep):
    write("POSCAR." + str(i), image, format="vasp")
