#! /usr/bin/env python
"""Program for structure optimization."""

from ase.io import read
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton


def main():
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    image_file = "initial.traj"
    fmax = 0.01
    steps = 1000
    traj = "initial.traj"
    log = "initial.log"

    # --------------------------------------------------------------------------
    # Load and optimize the image
    image = read(image_file)
    image.set_calculator(gen_calc())
    run_opt(image, fmax, steps, traj=traj, log=log)


def gen_calc():
    """Generate an EMT calculator."""
    return EMT()


def run_opt(image, fmax=0.01, steps=1000, traj="opt.traj", log="opt.log"):
    """Structure optimization using line-search BFGS algorithm."""
    opt_runner = QuasiNewton(image, trajectory=traj, logfile=log)
    opt_runner.run(fmax=fmax, steps=steps)


if __name__ == "__main__":
    main()
