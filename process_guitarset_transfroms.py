"""
The script in this file iterates through all of the .wav & .jam file pairs, computing transforms on the .wav files & performing procesing to the .jam files.  The resulting numpy arrays are uploaded to s3
"""
# !/usr/bin/env python3

import os
import numpy as np
import librosa
import boto3
from transcriptions.transcription_utils import jam_to_notes_matrix, annotation_audio_file_paths, notes_matrix_to_annotation
from transcriptions.transcription import Transcription


def process_wav_jam_pair(jam, wav, i):
    """
    This function performs the CQT transform on the wav file, & creates a matrix encoding of the annotation file and
    uploads the resulting numpy arrays to s3 :param jam: :param wav: :param i: :return:
    """

    bucket = 'dakobed-guitarset'
    s3 = boto3.client('s3')
    os.mkdir('data/dakobed-guitarset/fileID{}'.format(i))

    y, sr = librosa.load(wav)
    cqt_raw = librosa.core.cqt(y, sr=sr, n_bins=144, bins_per_octave=36, fmin=librosa.note_to_hz('C2'), norm=1)
    magphase_cqt = librosa.magphase(cqt_raw)
    cqt = magphase_cqt[0].T
    notes = jam_to_notes_matrix(jam)

    binary_annotation, multivariate_annotation = notes_matrix_to_annotation(notes, cqt.shape[0])

    for file, array, s3path in [
        ('data/dakobed-guitarset/fileID{}/cqt.npy'.format(i), cqt, 'fileID{}/cqt.npy'.format(i)),
        ('data/dakobed-guitarset/fileID{}/binary_annotation.npy'.format(i), binary_annotation,
         'fileID{}/binary_annotation.npy'.format(i)),
        ('data/dakobed-guitarset/fileID{}/multivariate_annotation.npy'.format(i), multivariate_annotation,
         'fileID{}/multivariate_annotation.npy'.format(i))]:
        np.save(file, arr=array)
        with open(file, "rb") as f:
            s3.upload_fileobj(f, bucket, s3path)
    with open(wav, "rb") as f:
        s3.upload_fileobj(f, bucket, "fileID{}/audio.wav".format(i))


def insert_guitarset_data_dynamo(dynamoDB, title, fileID):
    """
    This function uploads an item to dynamodb
    :param dynamoDB:
    :param title:
    :param fileID:
    :return:
    """

    print("Processing fileID {}".format(fileID))
    try:
        dynamoDB.put_item(
            TableName="DakobedGuitarSet",
            Item={
                "fileID": {"S": str(fileID)},
                "title": {"S": title}
            }
        )
    except Exception as e:
        print(e)


def save_transforms_and_annotations():
    """
    Itterate through the pairs
    """
    dynamoDB = boto3.client('dynamodb', region_name='us-west-2')
    files = annotation_audio_file_paths()
    for fileID in range(len(files)):
        wav = files[fileID][0]
        jam = files[fileID][1]
        title = wav.split('/')[-1].split('.wav')[0][3:-4]
        # insert_guitarset_data_dynamo(dynamoDB, title, fileID)
        process_wav_jam_pair(jam, wav, fileID)
        # tab = Transcription(wav, jam, fileID)
        print("Processed filepair " + str(fileID))


if __name__ == '__main__':
    save_transforms_and_annotations()
