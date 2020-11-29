import numpy as np


def load_transform(fileID):
    cqt = np.load('data/dakobed-guitarset/fileID{}/cqt.npy'.format(fileID))
    return cqt


def welford_files(file_range=260, frequency_bins = 144):
    '''
    This method will compute the variance and the mean over a range of arrays which are loaded in files
    :param file_range:  Compute the variance over the first file_range number of files for each column in the spectograms
    :param cqt:
    :return:THe runn
    '''
    n = 0
    delta = np.zeros(frequency_bins)
    mean = np.zeros(frequency_bins)
    M2 = np.zeros(frequency_bins)
    fileids = set(range(file_range))
    # fileids.remove(835)
    while fileids:
        id_ = fileids.pop()
        print("file {}".format(id_))
        data = load_transform(id_)
        for row in data:
            n = n + 1
            delta = row - mean
            mean = mean + delta / n
            M2 = M2 + delta * (row - mean)
    welford_variance = M2 / (n - 1)
    return welford_variance, mean