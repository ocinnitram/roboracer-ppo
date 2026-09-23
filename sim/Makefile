# make: build/libroboracer.so | make render: build/librender.so (fetches raylib) | make test | make clean
CC      ?= cc
CFLAGS   = -std=gnu11 -O3 -march=native -fPIC -fopenmp \
           -Wall -Wextra -Wno-unused -Wno-sign-compare
SRC      = csrc/binding.c csrc/shim.c
HDR      = csrc/roboracer.h csrc/tyre_dr_basin.h csrc/vecenv.h

TEST_CFLAGS = -std=gnu11 -O2 -g -Wall -Wextra -Wno-unused -Wno-sign-compare -Icsrc -Itests
TESTS    = tests/test_dynamics tests/test_env tests/test_edt

MAP      = lab-20260705
TRACK    = ../tracks/$(MAP)
BLOB     = $(TRACK)/$(MAP).blob
PYTHON   = cd .. && uv run --frozen python

RAYLIB     = .deps/raylib-5.5_linux_amd64
RAYLIB_URL = https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_linux_amd64.tar.gz

.PHONY: all render test clean
all: build/libroboracer.so
render: build/librender.so

build/libroboracer.so: $(SRC) $(HDR)
	@mkdir -p build
	$(CC) $(CFLAGS) -Icsrc -shared $(SRC) -lm -lpthread -o $@

$(RAYLIB)/lib/libraylib.a:
	@mkdir -p .deps
	curl -fsSL $(RAYLIB_URL) | tar xz -C .deps

build/librender.so: csrc/render.c $(HDR) $(RAYLIB)/lib/libraylib.a
	@mkdir -p build
	$(CC) $(filter-out -fopenmp,$(CFLAGS)) -Icsrc -I$(RAYLIB)/include -shared $< $(RAYLIB)/lib/libraylib.a -lm -lpthread -ldl -o $@

$(BLOB): $(TRACK)/raceline.csv $(TRACK)/$(MAP).yaml $(TRACK)/$(MAP).pgm
	$(PYTHON) -m sim.track $(MAP)

tests/fixtures.h: tests/reference.py
	$(PYTHON) -m sim.tests.reference

tests/fixtures_edt.h: tests/reference_edt.py $(BLOB)
	$(PYTHON) -m sim.tests.reference_edt

tests/test_dynamics: tests/test_dynamics.c tests/fixtures.h $(HDR)
	$(CC) $(TEST_CFLAGS) $< -lm -o $@

tests/test_env: tests/test_env.c tests/fixtures_env.h $(BLOB) $(HDR)
	$(CC) $(TEST_CFLAGS) $< -lm -o $@

tests/test_edt: tests/test_edt.c tests/fixtures_edt.h $(HDR)
	$(CC) $(TEST_CFLAGS) $< -lm -o $@

test: all $(TESTS)
	./tests/test_dynamics
	./tests/test_env
	./tests/test_edt

clean:
	rm -rf build $(TESTS) tests/fixtures.h tests/fixtures_edt.h
