from __future__ import print_function, division
from warnings import warn, filterwarnings

from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import h5py
import random
import sys

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, Reshape, Dropout
from keras.utils import plot_model

from nilmtk.utils import find_nearest
from nilmtk.feature_detectors import cluster
from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

class DAEDisaggregator(Disaggregator):
    '''Denoising Autoencoder disaggregator from Neural NILM
    https://arxiv.org/pdf/1507.06594.pdf

    Attributes
    ----------
    model : keras Sequential model
    sequence_length : the size of window to use on the aggregate data
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''

    def __init__(self, sequence_length):
        '''Initialize disaggregator

        Parameters
        ----------
        sequence_length : the size of window to use on the aggregate data
        meter : a nilmtk.ElecMeter meter of the appliance to be disaggregated
        '''
        self.MODEL_NAME = "AUTOENCODER"
        self.mmax = None
        self.sequence_length = sequence_length
        self.MIN_CHUNK_LENGTH = sequence_length
        #self.model = self._create_model(self.sequence_length)

    def train(self, mains, meter,apliance , epochs=1, batch_size=16, **load_kwargs):
        '''Train

        Parameters
        ----------
        mains : a nilmtk.ElecMeter object for the aggregate data
        meter : a nilmtk.ElecMeter object for the meter data
        epochs : number of epochs to train
        **load_kwargs : keyword arguments passed to `meter.power_series()`
        '''

        filename = apliance
        counter = 1
        main_power_series = mains.power_series(**load_kwargs)
        meter_power_series = meter.power_series(**load_kwargs)

        # Train chunks
        run = True
        mainchunk = next(main_power_series)
        meterchunk = next(meter_power_series)
        if self.mmax == None:
            self.mmax = mainchunk.max()

        while(run):
            mainchunk = self._normalize(mainchunk, self.mmax)
            meterchunk = self._normalize(meterchunk, self.mmax)
            filename = filename +str(counter)

            self.train_on_chunk(mainchunk, meterchunk, epochs, batch_size,filename)
            try:
                mainchunk = next(main_power_series)
                meterchunk = next(meter_power_series)
                counter +=1
            except:
                run = False

    def train_on_chunk(self, mainchunk, meterchunk, epochs, batch_size, filename):
        '''Train using only one chunk

        Parameters
        ----------
        mainchunk : chunk of site meter
        meterchunk : chunk of appliance
        epochs : number of epochs for training
        '''
        s = self.sequence_length
        #up_limit =  min(len(mainchunk), len(meterchunk))
        #down_limit =  max(len(mainchunk), len(meterchunk))

        # Replace NaNs with 0s
        mainchunk.fillna(0, inplace=True)
        meterchunk.fillna(0, inplace=True)
        ix = mainchunk.index.intersection(meterchunk.index)
        mainchunk = mainchunk[ix]
        meterchunk = meterchunk[ix]

        # Create array of batches
        #additional = s - ((up_limit-down_limit) % s)
        additional = s - (len(ix) % s)
        X_batch = np.append(mainchunk, np.zeros(additional))
        Y_batch = np.append(meterchunk, np.zeros(additional))

        X_batch = np.reshape(X_batch, (int(len(X_batch) / s), s))
        Y_batch = np.reshape(Y_batch, (int(len(Y_batch) / s), s))

        #SAVE BATCH----
        np.savetxt(filename,X_batch ,delimiter=" ")
        np.savetxt(filename+"-labels", Y_batch, delimiter=" ")

        #self.model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=epochs, shuffle=True)


    def _normalize(self, chunk, mmax):
        '''Normalizes timeseries

        Parameters
        ----------
        chunk : the timeseries to normalize
        max : max value of the powerseries

        Returns: Normalized timeseries
        '''
        tchunk = chunk / mmax
        return tchunk

    def _denormalize(self, chunk, mmax):
        '''Deormalizes timeseries
        Note: This is not entirely correct

        Parameters
        ----------
        chunk : the timeseries to denormalize
        max : max value used for normalization

        Returns: Denormalized timeseries
        '''
        tchunk = chunk * mmax
        return tchunk

