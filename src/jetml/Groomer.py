# This file is part of GroomRL by S. Carrazza and F. A. Dreyer

from .JetTree import JetTree
import numpy as np
import math
from abc import ABC, abstractmethod

#======================================================================
class AbstractGroomer(ABC):
    """AbstractGroomer class."""

    #----------------------------------------------------------------------
    def __call__(self, jet, returnTree = True):
        """Apply the groomer after casting the jet to a JetTree, and return groomed momenta."""
        # TODO: replace result by reclustered jet of all remaining constituents.
        if type(jet)==JetTree:
            tree = jet
        else:
            tree = JetTree(jet)
        self._groom(tree)
        if returnTree:
            return tree
        return tree.jet()

    #----------------------------------------------------------------------
    @abstractmethod
    def _groom(self, tree):
        pass

#======================================================================
class RSD(AbstractGroomer):
    """RSD applies Recursive Soft Drop grooming to a JetTree."""

    #----------------------------------------------------------------------
    def __init__(self, zcut=0.5, beta=1., R0=0.4):
        """Initialize RSD with its parameters."""
        self.zcut = zcut
        self.beta = beta
        self.R0   = R0

    #----------------------------------------------------------------------
    def _groom(self, tree):
        """Apply RSD grooming to a jet."""
        if not tree.lundCoord:
            # current node has no subjets => no grooming
            return
        state=tree.state()
        if not state.size>0:
            # current node has no subjets => no grooming
            return
        # check the SD condition
        z     = math.exp(state[0])
        delta = math.exp(state[1])
        remove_soft = (z < self.zcut * math.pow(delta/self.R0, self.beta))
        if remove_soft:
            # call internal method to remove soft branch and replace
            # current tree node with harder branch
            tree.remove_soft()
            # now we groom the new tree, since both nodes have been changed
            self._groom(tree)
        else:
            # if we don't groom the current soft branch, then continue
            # iterating on both subjets
            if tree.harder:
                self._groom(tree.harder)
            if tree.softer:
                self._groom(tree.softer)
