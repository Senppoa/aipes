"""
Module for managing datasets for training and validating Amp calculators.

Classes
-------
Dataset:
    Class for managing the dataset for cross-validation and bootstrap
    resampling techniques.
"""

from copy import deepcopy
from random import shuffle, randint

from ase.io import read


class Dataset(object):
    """
    Class for managing the dataset for either cross-validation or bootstrap
    resampling techniques.

    For cross-validation, firstly group the dataset into subgroups and then
    select one as the validation set and others as the training set.

    For bootstrap resampling, simply select num_sample data points from the
    dataset.

    Methods
    -------
    load:
        Load ASE trajectory file.
    group:
        Group the data loaded from trajectory file into subgroups for
        further selection.
    select:
        Select one group from grouped data as the validation dataset
        and others as the training dataset.
    select_bootstrap:
        Select data with replacement for bootstrap statistics.

    Parameters
    ----------
    trajname: str
        File name of the ASE trajectory file containing reference energies
        and forces. If not specified, the trajectory may also be loaded
        via the 'load' method later.
    """

    def __init__(self, trajname=None):
        self.rawdata = None
        self.grouped_data = None
        self.ngroup = 0
        if trajname is not None:
            self.load(trajname)

    def load(self, trajname=None):
        """Load ASE trajectory file."""
        if trajname is None:
            raise ValueError("Invalid trajectory file name!")
        if self.rawdata is not None:
            print("You are reloading a dataset. Regroup it before cross "
                  "validation.")
        self.rawdata = read(trajname, index=":")

    def group(self, ngroup=1, rand=True):
        """Group rawdata into subgroups for cross validation.

        Parameters
        ----------
        ngroup: integer
            Number of subgroups.
        rand: boolean
            Determines whether to shuffle rawdata (in fact a copy) before
            grouping.

        Returns
        -------
        None
        """
        if self.rawdata is None:
            raise IOError("Trajectory file not loaded!")
        if rand is True:
            worklist = deepcopy(self.rawdata)
            shuffle(worklist)
        else:
            worklist = self.rawdata
        self.grouped_data = [[worklist[j] for j in range(len(worklist))
                              if j % ngroup == i] for i in range(ngroup)]
        self.ngroup = ngroup

    def select(self, igroup=0):
        """Select the ith group for validation and the others for training.

        Parameters
        ----------
        igroup: integer
            Id of the subgroup chosen as the training dataset.

        Returns
        -------
        train_set: list of ASE atoms object
            Training dataset.
        valid_set: list of ASE atoms object
            Validation dataset.

        CAUTION
        -------
        Call the 'group' method before calling this method.
        """
        if self.grouped_data is None:
            raise RuntimeError("Group the dataset before cross validation!")
        valid_set = self.grouped_data[igroup]
        train_set = [data for group in self.grouped_data for data in group
                     if group is not valid_set]
        return train_set, valid_set

    def select_bootstrap(self, num_sample=1):
        """Create N samples from rawdata for bootstrap resampling.

        Parameters
        ----------
        num_sample: integer
            Number of samplings to be selected.

        Returns
        -------
        sampling_set: list of ASE objects
            Selected bootstrap samplings.
        """
        if self.rawdata is None:
            raise IOError("Trajectory file not loaded!")
        ndata = len(self.rawdata)
        sampling_set = [[self.rawdata[randint(0, ndata - 1)]
                         for j in range(ndata)] for i in range(num_sample)]
        return sampling_set
