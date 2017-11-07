"""
If forces are to be included in the objection function, be sure to use force-
consistent energies instead of the energies extrapolated to 0 Kelvin. However,
not all the calculators return force-consistent energies directly. This module
provides derived calculators which returns force-consistent energies regardless
of whether 'force_consistent' has been set to True or not.

Available modules
-----------------
vasp:
    Derived VASP calculator.
"""
