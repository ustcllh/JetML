CXX=g++
SWIG=swig
CXX_FLAGS= -std=c++11 -fPIC -fno-lto

PYTHON_CONFIG=python3-config
PYTHON_INCLUDE=`python3-config --includes`
PYTHON_LDFLAGS=`python3-config --ldflags`
PYTHON_LIB=`python3-config --libs`

FASTJET_CONFIG=/workspace/fastjet/bin/fastjet-config
FASTJET_LIB=`$(FASTJET_CONFIG) --libs`
FASTJET_INCLUDE=`$(FASTJET_CONFIG) --cxxflags`

SRC=src/JetTree

all: $(SRC)/JetTree_wrap.cxx\
	$(SRC)/JetTree.o\
	$(SRC)/JetTree_wrap.o\
	$(SRC)/_JetTree.so\
	loss


$(SRC)/JetTree_wrap.cxx: $(SRC)/JetTree.i
	$(SWIG) -c++ -python $<

$(SRC)/JetTree.o: $(SRC)/JetTree.cxx
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(FASTJET_INCLUDE) $(PYTHON_INCLUDE) -I$(SRC)

$(SRC)/JetTree_wrap.o: $(SRC)/JetTree_wrap.cxx
	$(CXX) $(CXX_FLAGS) -c $< -o $@ $(FASTJET_INCLUDE) $(PYTHON_INCLUDE)

$(SRC)/_JetTree.so: $(SRC)/JetTree.o $(SRC)/JetTree_wrap.o
	$(CXX) $(CXX_FLAGS) $^ -shared -o $@ $(PYTHON_LDFLAGS) $(FASTJET_LIB) $(PYTHON_LIB) -lstdc++

loss:
	mkdir -p loss

clean:
	rm $(SRC)/JetTree.py
	rm $(SRC)/JetTree_wrap.cxx
	rm $(SRC)/*.o
	rm $(SRC)/*.so
