
CC=gcc
NVCC=nvcc

CFLAGS=-std=c99 -pedantic -Wall -O2 -march=native
CUFLAGS=

LDFLAGS=-lm
CU_LDFLAGS=-lcuda -lcudart -lstdc++

SOURCES=image_io.c main.c

CPU_SOURCES=blur_cpu.c
GPU_SOURCES=blur_gpu.cu

OBJECTS=$(SOURCES:.c=.o)
CPU_OBJECTS=$(CPU_SOURCES:.c=.o)
GPU_OBJECTS=$(GPU_SOURCES:.cu=.o)

CPU_EXECUTABLE=soupify
GPU_EXECUTABLE=soupify-gpu

IMAGEMAGICK_OPTS=$(shell pkg-config --cflags --libs MagickWand)



all:
	@echo "Make cpu or make gpu?"


cpu: $(SOURCES) $(CPU_SOURCES) $(CPU_EXECUTABLE)


$(CPU_EXECUTABLE): $(OBJECTS) $(CPU_OBJECTS)
	$(CC) $(LDFLAGS) $(IMAGEMAGICK_OPTS) $(OBJECTS) $(CPU_OBJECTS) -o $@


%.o : %.c
	$(CC) $(IMAGEMAGICK_OPTS) $(CFLAGS) -c $<


gpu: $(SOURCES) $(GPU_SOURCES) $(GPU_EXECUTABLE)


$(GPU_EXECUTABLE): $(OBJECTS) $(GPU_OBJECTS)
	$(CC) $(LDFLAGS) $(IMAGEMAGICK_OPTS) $(CU_LDFLAGS) $(OBJECTS) $(GPU_OBJECTS) -o $@


%.o : %.cu
	$(NVCC) $(CUFLAGS) -c $<


clean:
	rm -f $(OBJECTS) $(CPU_OBJECTS) $(GPU_OBJECTS) $(CPU_EXECUTABLE) $(GPU_EXECUTABLE)
