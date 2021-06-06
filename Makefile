CC=nvcc
RM=rm -frd
CFLAGS=-std=c++14 -Wall -Werror -Wextra -Wfatal-errors -Wpedantic -pedantic-errors cross-execution-space-call
LDFLAGS=
LIBS=-lm
SOURCES=main.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

debug: CFLAGS+=-Og -g
debug: all
release: CFLAGS+=-DNDEBUG -O3 -flto -s
release: LDFLAGS+=-O3 -flto -s
release: all

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LDLIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) $(OBJECTS) $(EXECUTABLE)
