CC=nvcc
RM=rm -frd
CFLAGS=-std=c++14 -Werror \
	cross-execution-space-call,deprecated-declarations,reorder
LDFLAGS=
LIBS=-lm
SOURCES=main.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

debug: CFLAGS+=-g
debug: all
release: CFLAGS+=-DNDEBUG -O3
release: LDFLAGS+=-O3
release: all

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

$(OBJECTS): $(SOURCES)
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)
