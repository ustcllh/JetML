CXX=/usr/local/Cellar/gcc/9.1.0/bin/g++-9
CXXFLAGS=-g -std=c++17 -Wall -pedantic

ROOTFLAG=-pthread -m64 -I/usr/local/Cellar/root/6.18.00/include/root -L/usr/local/Cellar/root/6.18.00/lib/root

ROOTLIBS=-lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lpthread -lm -ldl

all: plot ang_plot

plot: plot.C
	$(CXX) $(CXXFLAGS) $(ROOTFLAG) -o $@ $< $(ROOTLIBS)

ang_plot: ang_plot.C
	$(CXX) $(CXXFLAGS) $(ROOTFLAG) -o $@ $< $(ROOTLIBS)

.PHONY:
	clean
	all

clean:
	rm -rf plot
	rm -rf ang_plot
