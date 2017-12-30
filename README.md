# aipes

## 1. Introduction

AIPES is a library that combines the subroutines provided by the atomic
simulation environment (ASE), such as the dimer method, nudged elastic band
(NEB), molecular dynamics (MD), and global optimization (GO), with machine
learning techniques. The key idea is to predict the potential energy surface
(PES) with machine learning based calculator trained from reference data
generated from first principles calculations. The predicted PES are then
validated using first-principles based reference calculator, and new train data
will be added to the training dataset to gradually improve the prediction
quality.

Currently only NEB has been implemented, and the only supported machine learning
calculator is Amp. There's still much work to do.

## 2. Installation

For installation instructions, see doc/install.md.


## 3. Tutorials

The examples under the 'test' directory act as benchmarks as well as tutorials.
See test/tutorials.md for more instructions.

Homepage of ASE: https://wiki.fysik.dtu.dk/ase/

Homepage of Amp: https://bitbucket.org/andrewpeterson/amp
