"""This file contains a test of the welford_files method, which calculates a running mean & variance over all
dimensions of the spectograms loaded.

"""


from pytest import approx
from transcriptions.welford import welford_files
import numpy as np


def test_welford():
    '''
    Test that the calculated variances are accurate
    :return:
    '''
    spec1 = np.load('../train/fileID0/cqt.npy')
    spec2 = np.load('../train/fileID1/cqt.npy')
    concat = np.concatenate((spec1, spec2), axis=0)

    concat_variance = concat.var(axis=0)
    concat_mean = concat.mean(axis=0)
    var, mean = welford_files(2)
    assert var == approx(concat_variance, rel=1e-1)
    assert mean == approx(concat_mean, rel=1e-1)