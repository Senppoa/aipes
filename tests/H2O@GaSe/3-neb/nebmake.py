#! /usr/bin/env python
"""Prepare directories and POSCAR files for VTST-NEB Calculation."""

import os
import sys

from ase.io import read, write


mep = read(sys.argv[1], index=":")

for i, image in enumerate(mep):
    dir_name = str("%02d" % i)
    os.mkdir(dir_name)
    write(dir_name + "/POSCAR", image, format="vasp")
