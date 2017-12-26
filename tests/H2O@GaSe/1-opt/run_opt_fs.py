#! /usr/bin/env python
"""Program for structure optimization."""

import os

from ase.io import read
from aipes.calculators.vasp import VaspFC as Vasp
from ase.optimize import QuasiNewton


def main():
    # --------------------------------------------------------------------------
    # Declare controlling parameters
    image_file = "POSCAR.fs"
    fmax = 0.05
    steps = 5000
    traj = "final.traj"
    log = "final.log"

    # --------------------------------------------------------------------------
    # Load and optimize the image
    image = read(image_file)
    image.set_calculator(gen_calc())
    run_opt(image, fmax, steps, traj=traj, log=log)


def gen_calc():
    """Setup runtime environment for VASP and generate a calculator."""
    os.environ["VASP_PP_PATH"] = "/home/yhli/soft/pp.v54"
    os.environ["VASP_COMMAND"] = "mpirun -np 12 vasp_std &> vasp.log"
    calc = Vasp(restart=False, xc="PBE", kpts=[3, 3, 1], gamma=True,
                prec="Accurate", 
                ispin=2, lvdw=True,
                encut=400, nelm=60, ediff=1.e-4, nelmin=6, lreal="Auto",
                nsw=0, isym=0, ismear=0, sigma=0.05)
    return calc


def run_opt(image, fmax=0.01, steps=1000, traj="opt.traj", log="opt.log"):
    """Structure optimization using line-search BFGS algorithm."""
    opt_runner = QuasiNewton(image, trajectory=traj, logfile=log)
    opt_runner.run(fmax=fmax, steps=steps)


if __name__ == "__main__":
    main()
