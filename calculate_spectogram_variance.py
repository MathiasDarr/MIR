"""This script computes the mean & the variance of all the spectograms.  Since the entire dataset does not fit into
memory, the welford method for computing the variance is used.  The resulting numpy arrays is uploaded to s3 """
# !/usr/bin/env python3

import boto3
from transcriptions.welford import welford_files
import numpy as np


var, mean = welford_files()
np.save(arr=var, file='../data/guitarset-var.npy')
np.save(arr=mean, file='../data/guitarset-mean.npy')
bucket = 'dakobed-guitarset'
s3 = boto3.client('s3')

with open('../data/guitarset-var.npy', "rb") as f:
    a = s3.upload_fileobj(f, bucket, 'guitarset-var.npy')
with open('../data/guitarset-mean.npy', "rb") as f:
    s3.upload_fileobj(f, bucket, 'guitarset-mean.npy')
