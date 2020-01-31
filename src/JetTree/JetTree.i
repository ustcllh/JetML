%module JetTree


%{
#define SWIG
#include "JetTree.h"
%}

%include "std_vector.i"

namespace std {
    %template(PseudoJetVec) vector<fastjet::PseudoJet>;
    %template(IntVec) vector<int>;
    %template(DoubleVec) vector<double>;
};


%include "JetTree.h"
