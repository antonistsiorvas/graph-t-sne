CC     = gcc
CFLAGS = -O3 -march=native -Wall
MATIO  = $(shell pkg-config --cflags --libs matio)

seq: src/seq.c src/tsne.c src/save.c src/data.c src/tsne.h
	$(CC) $(CFLAGS) -o $@ src/seq.c src/tsne.c src/save.c src/data.c $(MATIO) -lm

count_terms: src/count_terms.c src/tsne.c src/data.c src/tsne.h
	$(CC) $(CFLAGS) -o $@ src/count_terms.c src/tsne.c src/data.c $(MATIO) -lm

clean:
	rm -f seq count_terms
.PHONY: clean
