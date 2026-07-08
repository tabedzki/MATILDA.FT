# Copyright (c) 2023 University of Pennsylvania
# Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#
####################################################
# Modify the line below to point to the CUDA library
CUDA_LOC = /usr/local/cuda




CC = ${CUDA_LOC}/bin/nvcc
# GIT_VERSION := $(shell git describe --abbrev=9 --dirty --always --tags)
# GIT_DIRECTORY := $(shell git rev-parse --show-toplevel)

# EIGEN_LOC = ../include/eigen-3.4.0

####################################################
# GPU architecture selection
#
#   make                        - detect the GPU(s) in this machine and build
#                                 for exactly those; if none are found, build
#                                 the portable fat binary below
#   make CUDA_ARCHS=fat         - fat binary for the FAT_ARCHS list + PTX for
#                                 the newest one (future GPUs JIT-compile it)
#   make CUDA_ARCHS=all         - every architecture this nvcc supports
#   make CUDA_ARCHS="70 80"     - explicit list of compute capabilities
#
# Requested architectures that the installed CUDA toolkit cannot target are
# dropped with a note (e.g. sm_37/K80 needs CUDA <= 11.x), so the same
# makefile works across toolkit versions without editing.
CUDA_ARCHS ?= detect

# Curated portable set: K80 (37), P100 (60), V100 (70), A100 (80), H100 (90).
# In-between cards run the nearest lower entry of the same major version
# (e.g. sm_86 runs the sm_80 cubin); newer ones JIT the embedded PTX.
FAT_ARCHS = 37 60 70 80 90

# Everything the installed nvcc can generate code for
NVCC_ARCHS := $(patsubst compute_%,%,$(shell ${CC} --list-gpu-arch 2>/dev/null))

ifeq (${CUDA_ARCHS},detect)
  REQUESTED := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d . | sort -un)
  ifeq (${REQUESTED},)
    $(info makefile: no local GPU detected, building fat binary for: ${FAT_ARCHS})
    REQUESTED := ${FAT_ARCHS}
  endif
else ifeq (${CUDA_ARCHS},fat)
  REQUESTED := ${FAT_ARCHS}
else ifeq (${CUDA_ARCHS},all)
  REQUESTED := ${NVCC_ARCHS}
else
  REQUESTED := ${CUDA_ARCHS}
endif

SELECTED := $(filter ${NVCC_ARCHS},${REQUESTED})
DROPPED  := $(filter-out ${NVCC_ARCHS},${REQUESTED})
ifneq (${DROPPED},)
  $(info makefile: dropping architecture(s) not supported by this CUDA toolkit: ${DROPPED} (supported: ${NVCC_ARCHS}))
endif
ifeq (${SELECTED},)
  ifeq (${CUDA_ARCHS},detect)
    # e.g. a K80 under CUDA 12: can't target the local GPU, so at least
    # produce the portable binary instead of failing
    $(info makefile: local GPU not supported by this toolkit, building fat binary instead)
    SELECTED := $(filter ${NVCC_ARCHS},${FAT_ARCHS})
  else
    $(error no requested GPU architecture is supported by ${CC}; supported: ${NVCC_ARCHS})
  endif
endif

# cubins for every selected arch + PTX for the newest, so GPUs newer than
# this toolkit can still run the binary via JIT compilation
NEWEST_ARCH := $(lastword $(shell printf '%s\n' ${SELECTED} | sort -n))
ARCH_FLAGS  := $(foreach a,${SELECTED},-gencode arch=compute_${a},code=sm_${a}) \
               -gencode arch=compute_${NEWEST_ARCH},code=compute_${NEWEST_ARCH}
####################################################

CFLAGS = -std=c++14 ${ARCH_FLAGS} -rdc=true -O3 -I${CUDA_LOC}/include -Wno-deprecated-gpu-targets
CPPFLAGS = --expt-relaxed-constexpr
# CFLAGS = -std=c++14 -arch=sm_70 -rdc=true -O3 -I${CUDA_LOC}/include -I${EIGEN_LOC} -Wno-deprecated-gpu-targets
# CPPFLAGS = -I${EIGEN_LOC} --expt-relaxed-constexpr

#CFLAGS = -std=c++11 -rdc=true -arch=sm_37 -dlto -O3 -I${CUDA_LOC}/include -lineinfo -g 

LIBS = -lm -lcudart -lcufft -L${CUDA_LOC}/lib

       
SRCS = 	Box.cu main.cu die.cu random.cu device_utils.cu \
	FTS_Box.cu fts_species.cu fts_potential.cu fts_potential_helfand.cu fts_potential_flory.cu \
	fts_molecule.cu fts_molecule_linear.cu fts_updateSchemes.cu fts_potential_edwards.cu \
	fts_potential_incompress.cu fts_moleculeHParticle.cu fts_potentialParticle.cu \
	fts_make_CLnoise.cu \
	FTS_BoxFindSpinodal.cu \
	PS_Box.cu PS_Box_initialization.cu PS_Box_gsdUtils.cu PS_Box_moleculeMakers.cu \
	ps_species.cu ps_grid_utils.cu ps_groups.cu \
	ps_integrator.cu ps_integratorGJF.cu ps_integratorEM.cu ps_integratorVV.cu \
	ps_deviceBonds.cu ps_potential.cu ps_potentialGaussian.cu ps_deviceAngles.cu \
	ps_potentialCharges.cu ps_potentialMaierSaupe.cu ps_potentialLangevin.cu \
	ps_potentialBias.cu ps_potentialErf2.cu ps_potentialErfG.cu ps_potentialBias_lc.cu \
	ps_potentialDPD.cu ps_potentialWall.cu \
	ps_neighborList.cu ps_compute.cu ps_compute_sk.cu \

# Box_fftWrappers.cu
# array_utils.cu device_array_utils.cu device_bonds.cu device_grid_utils.cu device_utils.cu  field_component.cu
# bonds.cu calc_properties.cu config_utils.cu cuda_random_posits.cu \
# device_array_utils.cu device_bonds.cu device_comm_utils.cu Compute_avg_rho.cu \
# device_grid_utils.cu device_utils.cu  field_component.cu forces.cu \
# initialize.cu integ_utils.cu io_utils.cu main.cu potential.cu \
# potential_bias_field.cu potential_gaussian.cu pbc_utils.cu read_input.cu \
# potential_erf.cu reduce_utils.cu update_potentials.cu device_GJF_integrator.cu \
# potential_charges.cu potential_gaussian_erf.cu \
# Compute.cu angles.cu device_angles.cu random.cu grid_utils.cu \
# Extraforce.cu die.cu \
# Extraforce_dpd.cu Extraforce_langevin.cu Extraforce_lewis.cu Extraforce_midpush.cu Extraforce_wall.cu \
# group.cu \
# group_id.cu group_region.cu group_type.cu\
# integrator.cu \
# integrator_VV.cu integrator_EM.cu integrator_GJF.cu\
# nlist.cu \
# nlist_half_distance.cu nlist_bonding.cu \
# Compute_widom.cu Compute_avg_sk.cu Compute_sk.cu Compute_chempot.cu\
# tensor_potential.cu tensor_potential_MaierSaupe.cu device_ms_utils.cu \


SRCS_C = gsd.c
#SRCS_CPP = tensor_potential_MaierSaupe_eigenvalue.cpp

OBJFOLDER := objects
OBJS_MAKE  	  = $(addprefix $(OBJFOLDER)/, ${OBJS})
OBJS_MAKE_C   = $(addprefix $(OBJFOLDER)/, ${OBJS_C})
OBJS_MAKE_CPP = $(addprefix $(OBJFOLDER)/, ${OBJS_CPP})

OBJS     = $(SRCS:.cu=.cu.o)
OBJS_C   = $(SRCS_C:.c=.c.o)
OBJS_CPP = $(SRCS_CPP:.cpp=.cpp.o)

$(OBJFOLDER)/%.cu.o: %.cu git-version.h
	@mkdir -p $(dir $@)
	${CC} -g ${CFLAGS} -c $< -o $@

$(OBJFOLDER)/%.cpp.o: %.cpp git-version.h
	@mkdir -p $(dir $@)
	${CC} -g ${CFLAGS} ${CPPFLAGS} -c $< -o $@

$(OBJFOLDER)/%.c.o: %.c git-version.h
	@mkdir -p $(dir $@)
	${CC} -g ${CFLAGS} -c $< -o $@
	 
../matilda.ft: $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C)
	$(CC) $(CFLAGS) -o $@ $(OBJS_MAKE) $(OBJS_MAKE_CPP) $(OBJS_MAKE_C) $(LIBS) -g

git-version.h: 
	sed "s/@GIT_VERSION@/$(GIT_VERSION)/"  git-version.h.in > git-version.h

clean:
	rm -fr *.o objects
	rm -f matilda.ft ../matilda.ft
	rm -f git-version.h
	rm -f *~

