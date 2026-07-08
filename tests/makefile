# Copyright (c) 2026 University of Pennsylvania
# Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#
# Unit-test build for MATILDA.FT.
#   make            - build the matilda_tests binary
#   make test       - build and run the unit tests
#   make regression - run the golden-file regression suite (needs ../matilda.ft)
#   make clean      - remove test build artifacts
#
# Reuses the object files produced by src/makefile (everything except
# main.cu.o, whose main() and globals are replaced by support/test_main.cu).

CUDA_LOC ?= /usr/local/cuda
NVCC := $(CUDA_LOC)/bin/nvcc
CXX  := g++

SRC_DIR   := ../src
OBJ_DIR   := $(SRC_DIR)/objects
GTEST_DIR := external/googletest/googletest
BUILD     := build

# override with: make ARCH=sm_86 (keep in sync with the arch used for src/)
ARCH ?= sm_70

# Match src/makefile flags so device code links cleanly (-rdc=true, same arch).
NVCCFLAGS := -std=c++14 -arch=$(ARCH) -rdc=true -O2 -g \
             -I$(CUDA_LOC)/include -I$(SRC_DIR) -isystem $(GTEST_DIR)/include \
             --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
LIBS := -lm -lcudart -lcufft -L$(CUDA_LOC)/lib -Xcompiler -pthread

TEST_SRCS := support/test_main.cu $(wildcard unit/*.cu)
TEST_OBJS := $(addprefix $(BUILD)/, $(TEST_SRCS:.cu=.cu.o))

all: matilda_tests

# Keep the project objects up to date; src/makefile owns that build.
.PHONY: proj-objs
proj-objs:
	$(MAKE) -C $(SRC_DIR)

$(BUILD)/gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	@mkdir -p $(dir $@)
	$(CXX) -std=c++14 -isystem $(GTEST_DIR)/include -I$(GTEST_DIR) -pthread -c $< -o $@

$(BUILD)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

# Project objects are listed at link time (after proj-objs has built them),
# excluding main.cu.o which defines main() and the MAIN globals.
matilda_tests: proj-objs $(TEST_OBJS) $(BUILD)/gtest-all.o
	$(NVCC) $(NVCCFLAGS) -o $@ $(TEST_OBJS) $(BUILD)/gtest-all.o \
		$$(ls $(OBJ_DIR)/*.o | grep -v '/main\.cu\.o') $(LIBS)

.PHONY: test
test: matilda_tests
	./matilda_tests

.PHONY: regression
regression:
	cd regression && ./run_regression.sh

.PHONY: clean
clean:
	rm -rf $(BUILD) matilda_tests

-include $(TEST_OBJS:.o=.d)
