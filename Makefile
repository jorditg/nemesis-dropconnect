CC=g++
CFLAGS=-O3 --std=c++11 -Wall
LIBFLAGS=-lOpenCL -pthread

HEADERS=nn.hpp OpenCLKernels.hpp common.hpp mg.hpp mnist.hpp cli.hpp
SOURCES=main.cpp nn.cpp OpenCLKernels.cpp common.cpp mg.cpp mnist.cpp cli.cpp
EXECUTABLE=nn-opencl

all: $(EXECUTABLE)

nn-opencl: $(HEADERS) $(SOURCES) Makefile
		$(CC) $(CFLAGS) $(SOURCES) $(LIBFLAGS) -o$(EXECUTABLE)

clean:
	rm -f *.o *~ $(EXECUTABLE)

