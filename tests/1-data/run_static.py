#! /usr/bin/env python
"""Program for total energy and forces calculation."""

from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT


def main():
    struct = "images.traj"
 
    images = read(struct, index=":")
    for image in images:
        image.set_calculator(gen_calc())
        image.get_potential_energy(apply_constraint=False)
        image.get_forces(apply_constraint=False)
    
    write("train.traj", images)


def gen_calc():
    """Generate an EMT calculator."""
    return EMT()


if __name__ == "__main__":
    main()
