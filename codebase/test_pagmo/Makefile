CXX=g++
LIBS=-lpthread -lpython2.7 
LDIRS=-L$$BASE_DIR/pagmo/lib
IDIRS=-I/usr/include/python2.7 -I$$BASE_DIR/pagmo/include/ \
      -I/usr/include/eigen3
CXXFLAGS=-std=c++14 $(IDIRS)

default:
	$(CXX) test_pagmo.cpp -o test_pagmo $(CXXFLAGS) $(LIBS) $(LDIRS)

run:
	./test_pagmo

clean:
	rm -f test_pagmo
