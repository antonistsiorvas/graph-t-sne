# ./seq  data/<graph> [L] [direct]
CC     = gcc
CFLAGS = -O3 -march=native -Wall
MATIO  = $(shell pkg-config --cflags --libs matio)

COMMON = src/tsne.c src/driver.c src/save.c src/data.c
DEPS   = $(COMMON) src/tsne.h src/driver.h src/data.h

all: seq

seq: src/seq.c $(DEPS)
	$(CC) $(CFLAGS) -o $@ src/seq.c $(COMMON) $(MATIO) -lm

count_terms: src/count_terms.c src/tsne.c src/data.c src/tsne.h
	$(CC) $(CFLAGS) -o $@ src/count_terms.c src/tsne.c src/data.c $(MATIO) -lm

clean:
	rm -f seq count_terms
.PHONY: all clean
