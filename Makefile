CXX = g++

INSTALL_DIR = ${PWD}

CPP_FILES = $(wildcard src/*.cpp)
OBJ_FILES = $(patsubst src/%.cpp,obj/%.o,$(CPP_FILES)) 
TEST_FILES = $(wildcard test/cpp/*.cpp)
TEST_BIN = $(patsubst test/cpp/%.cpp,test/bin/%,$(TEST_FILES))

FLAGS = -Wall -O3 

SLIB = libblitzl1.so

all: $(SLIB) 

$(SLIB): $(OBJ_FILES) | lib
	$(CXX) $(FLAGS) -shared -o ${INSTALL_DIR}/lib/$@ $^

obj/%.o: src/%.cpp
	$(CXX) $(FLAGS) -fPIC -c -o $@ $<

$(OBJ_FILES): | obj

obj:
	mkdir -p obj

lib:
	mkdir -p lib

test: $(SLIB) $(TEST_BIN)

test/bin/%: test/cpp/%.cpp | test_bin
	$(CXX) $(FLAGS) -L${INSTALL_DIR}/lib -lblitzl1 -o $@ $<

test_bin:
	mkdir -p test/bin

clean:
	rm -rf obj
	rm -rf lib
	rm -rf test/bin
