CXX=g++
SWIG=swig

PYTHON_CONFIG=python3-config
PYTHON_INCLUDE=`python3-config --includes`
PYTHON_LIB=`python3-config --libs`

FASTJET_CONFIG=/workspace/fastjet/bin/fastjet-config
FASTJET_LIB=`$(FASTJET_CONFIG) --libs`
FASTJET_INCLUDE=`$(FASTJET_CONFIG) --cxxflags`

SRC=src/JetTree

all: $(SRC)/JetTree_wrap.cxx\
	$(SRC)/JetTree.o\
	$(SRC)/JetTree_wrap.o\
	$(SRC)/_JetTree.so


$(SRC)/JetTree_wrap.cxx: $(SRC)/JetTree.i
		$(SWIG) -c++ -python $<

$(SRC)/JetTree.o: $(SRC)/JetTree.cxx
		$(CXX) -fPIC -c $< -o $@ $(FASTJET_INCLUDE) $(PYTHON_INCLUDE) -I$(SRC)

$(SRC)/JetTree_wrap.o: $(SRC)/JetTree_wrap.cxx
		$(CXX) -fPIC -c $< -o $@ $(FASTJET_INCLUDE) $(PYTHON_INCLUDE)

$(SRC)/_JetTree.so: $(SRC)/JetTree.o $(SRC)/JetTree_wrap.o
		$(CXX) $^ -shared -o $@ $(FASTJET_LIB) $(PYTHON_LIB) -lstdc++

clean:
	rm $(SRC)/JetTree.py
	rm $(SRC)/JetTree_wrap.cxx
	rm $(SRC)/*.o
	rm $(SRC)/*.so
