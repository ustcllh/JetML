CXX=/usr/local/Cellar/gcc/9.2.0_1/bin/g++-9
CXXFLAGS=-g -std=c++17 -Wall -pedantic

ROOTFLAG=-pthread -m64 -I/usr/local/Cellar/root/6.18.04/include/root -L/usr/local/Cellar/root/6.18.04/lib/root

ROOTLIBS=-lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lpthread -lm -ldl

all: getRootTree

getRootTree: ./src/getRootTree.C
	$(CXX) $(CXXFLAGS) $(ROOTFLAG) -o $@ $< $(ROOTLIBS)

.PHONY:
	clean
	all

clean:
	rm -rf getRootTree
