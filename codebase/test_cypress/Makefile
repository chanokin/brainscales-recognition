CXX=g++
LIBS=-lcypress -lpthread -lpython2.7
LDIRS=-L$$BASE_DIR/cypress/lib
IDIRS=-I/usr/include/python2.7 -I$$BASE_DIR/cypress/include/
CXXFLAGS=-std=c++14 $(IDIRS)

default:
	$(CXX) test_cypress.cpp -o test_cypress $(CXXFLAGS) $(LIBS) $(LDIRS)

run:
	./test_cypress pynn.nest

clean:
	rm -f test_cypress
