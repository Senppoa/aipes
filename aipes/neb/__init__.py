"""
This package provides subroutines for NEB calculation based on machine learning
calculator with first principles correction.

Both serial and parallel versions are provided. Each module provides a function
named "run_aineb".

Available modules
-----------------
common:
    Shared subroutines.
child:
    Script to start child process for NEB calculation, called by the 'dynamic'
    module.
dynamic:
    Parallel version of 'run_aineb' based on mpi4py, with dynamic process
    management.
serial:
    Serial version of 'run_aineb'.
parallel:
    Parallel version of 'run_aineb' based on mpi4py.
"""
