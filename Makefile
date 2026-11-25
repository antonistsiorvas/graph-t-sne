# ./seq  data/<graph> [L] [direct]
# OMP_NUM_THREADS=4 ./omp  data/<graph>
CC     = gcc
CFLAGS = -O3 -march=native -Wall
MATIO  = $(shell pkg-config --cflags --libs matio)

COMMON = src/tsne.c src/driver.c src/save.c src/data.c
DEPS   = $(COMMON) src/tsne.h src/driver.h src/data.h

all: seq omp

seq: src/seq.c $(DEPS)
	$(CC) $(CFLAGS) -o $@ src/seq.c $(COMMON) $(MATIO) -lm

omp: src/omp.c $(DEPS)
	$(CC) $(CFLAGS) -fopenmp -o $@ src/omp.c $(COMMON) $(MATIO) -lm



count_terms: src/count_terms.c src/tsne.c src/data.c src/tsne.h
	$(CC) $(CFLAGS) -o $@ src/count_terms.c src/tsne.c src/data.c $(MATIO) -lm

clean:
	rm -f seq omp count_terms
.PHONY: all clean
