# For Linux or any machines with gcc compiler
CC = gcc
CFLAGS = -Wall -pedantic
#CFLAGS = -Wall -pedantic -D N=9    # inserts macro #define N 9

all: transpose 

clean:
	/bin/rm *.o

OBJ = 

transpose-cpu: transpose-cpu.o $(OBJ) 
	$(CC) $(CFLAGS) -o $@ $@.o $(OBJ) -lm 
	/bin/rm *.o

transpose: transpose.cu
	nvcc -o $@ $@.cu

