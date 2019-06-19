import torch
import torch.nn as nn
from reader import *
from feed import *
from classifier import *

datafile = '/workspace/samples/PythiaEventsTune14PtHat120_0.pu14'
"""
datafile = '/workspace/samples/jewel_1.pu14'
"""

rd = Reader(datafile)
event = rd.next_event()
jf = Jet_Finder()
jets = jf(event)

jc = Jet_Classifier('lstm')

# for jet in jets:
print(datafile)
jet = jets[0]
print(jet)
x = jc(jet)
print(x)
