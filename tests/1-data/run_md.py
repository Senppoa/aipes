#! /usr/bin/env python
"""Program for molecular dynamics."""

from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution as MBDist
from ase.md.verlet import VelocityVerlet


def main():
    struct = "images.traj"
    temp = 100
    dt = 2.0
    steps = 5
 
    images = read(struct, index=":")
    for index, image in enumerate(images):
        image.set_calculator(gen_calc())
        if index == 0:
            traj = Trajectory("train.traj", mode="w", atoms=image)
        else:
            traj = Trajectory("train.traj", mode="a", atoms=image)
        run_md(image, temp, dt, steps, traj)


def gen_calc():
    """Generate an EMT calculator."""
    return EMT()


def run_md(image, temp, dt=1.0, steps=1000, traj="md.traj", log="md.log"):
    """Constant NVE dynamics using velocity verlet algorithm."""
    MBDist(image, temp=temp*units.kB, force_temp=True)
    md_runner = VelocityVerlet(image, dt=dt*units.fs, trajectory=traj, logfile=log)
    md_runner.run(steps=steps)


if __name__ == "__main__":
    main()
