CC=nvcc
RM=rm -frd
CFLAGS=-std=c++14 -Werror \
	cross-execution-space-call,deprecated-declarations,reorder
LDFLAGS=-lcublas -lcusolver
LIBS=-lm
INCDIR=include
SRCDIR=src
OBJDIR=obj
BINDIR=bin
SOURCES=$(wildcard $(SRCDIR)/*.cu)
OBJECTS=$(SOURCES:$(SRCDIR)/.cu=$(OBJDIR)/.o)
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
	$(CC) -I$(INCDIR) -c $(CFLAGS) $< -o $@

clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)
