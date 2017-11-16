#! /usr/bin/env python
"""Program for molecular dynamics."""

import numpy as np

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBDist
from ase.md.verlet import VelocityVerlet


def main():
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    initial_file = "initial.traj"
    final_file = "final.traj"
    num_inter_images = 50
    temp = 100
    dt = 2.0
    steps = 5

    # --------------------------------------------------------------------------
    # Load initial and final images
    initial_image = read(initial_file, index=-1)
    final_image = read(final_file, index=-1)

    # Interpolate
    images = interpolate(initial_image, final_image, num_inter_images)

    # Run molecular dynamics from each image in images
    for index, image in enumerate(images):
        print("Dealing with image # %d." % index, flush=True)
        image.set_calculator(gen_calc())
        if index == 0:
            traj = Trajectory("md.traj", mode="w", atoms=image)
        else:
            traj = Trajectory("md.traj", mode="a", atoms=image)
        run_md(image, temp, dt, steps, traj)


def gen_calc():
    """Generate an EMT calculator."""
    return EMT()


def interpolate(initial_image, final_image, num_inter_images):
    """Interpolate N intermediate images between initial and final images."""
    displacement = final_image.get_positions() - initial_image.get_positions()
    images = []
    for ita in np.linspace(0.0, 1.0, num_inter_images+2):
        image_copy = initial_image.copy()
        image_copy.positions += ita * displacement
        images.append(image_copy)
    return images


def run_md(image, temp, dt=1.0, steps=1000, traj="md.traj", log="md.log"):
    """Constant NVE dynamics using velocity verlet algorithm."""
    MBDist(image, temp=temp*units.kB, force_temp=True)
    md_runner = VelocityVerlet(image, dt=dt*units.fs, trajectory=traj,
                               logfile=log)
    md_runner.run(steps=steps)


if __name__ == "__main__":
    main()
