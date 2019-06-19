import os, sys
fastjet_path = '/workspace/fastjet/lib/python3.7/site-packages/fastjet.py'
sys.path.append(os.path.dirname(fastjet_path))

import fastjet as fj
from JetTree import *
import numpy as np

class Jet_Transformer():
    """
    Transform pseudojet to trainable data fotmat
    sequential substructure variables for lstm model
    jet image for cnn model
    """
    def __init__(self, model='lstm'):
        self.model = model

    def __call__(self, jet):
        return self._transform_(jet)

    def _transform_(self, jet):
        if self.model == 'lstm':
            return self._trans_lstm_(jet)
        elif self.model == 'cnn':
            return self._trans_cnn_(jet)

    def _trans_lstm_(self, jet, dim=6):
        jet_tree = JetTree(jet)
        if jet_tree.lundCoord:
            jet_tree.lundCoord.change_dimension(dim)
        res = []
        while 1:
            if jet_tree.lundCoord:
                res.append([
                jet_tree.lundCoord.lnm,
                jet_tree.lundCoord.lnKt,
                jet_tree.lundCoord.lnz,
                jet_tree.lundCoord.lnDelta,
                jet_tree.lundCoord.lnKappa,
                jet_tree.lundCoord.psi
                ])
            jet_tree = jet_tree.harder
            if not jet_tree:
                break
        if res:
            res = np.stack([res])
        else:
            res = np.zeros((1, 1, dim))
        return res

    def _trans_cnn_(self, jet, R=0.4, size=33):
        jet_pixeliser = Jet_Pixeliser(R, size, 5)
        return jet_pixeliser(jet)

class Jet_Pixeliser:
    def __init__(self, R=0.4, size=33, channel=5):
        self.R = R
        self.size = size
        self.channel = channel

    def __call__(self, jet, R=0.4, size=33, channel=5):
        self.__init__(R, size, channel)
        _mat = np.zeros([1, self.channel, self.size, self.size] , dtype=np.float32)
        for p in jet.constituents():
            dx = p.eta() - jet.eta()
            dy = p.phi() - jet.phi()
            if abs(dx)>R or abs(dy)>R:
                continue
            idx = self._get_index_(dx, self.size, self.R)
            idy = self._get_index_(dy, self.size, self.R)

            _mat[0][0][idx][idy]=abs(p.px())
            _mat[0][1][idx][idy]=abs(p.py())
            _mat[0][2][idx][idy]=abs(p.pz())
            _mat[0][3][idx][idy]=abs(p.e())
            _mat[0][4][idx][idy]+=1
        return _mat

    @staticmethod
    def _get_index_(dr, width, R):
        ix=0
        if width%2 == 0:
            if dr>=0:
                 ix = int(0.5*dr*width/R) + int(width/2)
            else:
                 ix = int(0.5*dr*width/R) + int(width/2) - 1
        else:
            if dr>=0:
                 ix = int(0.5*dr*width/R + 0.5) + int(width/2)
            else:
                 ix = int(0.5*dr*width/R - 0.5) + int(width/2)
        return ix
