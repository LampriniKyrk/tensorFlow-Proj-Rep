from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from daedisaggregator import DAEDisaggregator


print("========== OPEN DATASETS ============")
dsPath = '/home/nana/Downloads/ukdale.h5'
train = DataSet(dsPath)
test = DataSet(dsPath)

#train.set_window(start="13-4-2013", end="1-1-2014")
train.set_window(start="13-4-2013", end="13-9-2013")
test.set_window(start="1-1-2014", end="30-3-2014")

train_building = 1
test_building = 1
sample_period = 6
meter_key = 'microwave'
train_elec = train.buildings[train_building].elec
test_elec = test.buildings[test_building].elec

train_meter = train_elec.submeters()[meter_key]
test_meter = test_elec.submeters()[meter_key]
train_mains = train_elec.mains()
test_mains = test_elec.mains()
dae = DAEDisaggregator(7200) #half day

start = time.time()
print("========== TRAIN ============")
epochs = 0
epochsPerCheckpoint = 1
totalCheckpoints = 1
for i in range(totalCheckpoints):
    print("CHECKPOINT {}".format(epochs))
    dae.train(train_mains, train_meter, apliance=meter_key ,epochs=epochsPerCheckpoint, sample_period=sample_period)
    epochs += epochsPerCheckpoint

