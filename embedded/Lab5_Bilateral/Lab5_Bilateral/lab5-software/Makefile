.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make exe"
	$(ECHO) "      Command to build executable."
	$(ECHO) ""
	$(ECHO) "  make run "
	$(ECHO) "      Command to run the application on FPGA."
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

# The default architecture to build is ARM 64-bit.
HOST_ARCH := aarch64
# Used only for ARM cross-compiling
SYSROOT := /home/alex/petalinux2019_2/sysroot/sysroots/aarch64-xilinx-linux/

# The GCC/C++ Compiler to use is included here, depending arch
include ./lib/utils.mk

# Compiler flags
CXXFLAGS += -lm -Wall -O3 -g -fopenmp

# Linker flags
ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

#Host C FILES
HOST_C_SRCS += filter.c
EXECUTABLE = filter

# System command utilities
CP = cp -rf
SFTP = scp
SSH = ssh

# Building Host
.PHONY: exe
exe: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_C_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_C_SRCS) -o '$@' $(LDFLAGS)

# The run command for the FPGA.
# It uploads the executable (filter), data files and run via ssh the application.
# OMP_NUM_THREADS=N states the N number of threads to be used. 4 by default
# You can change N as you like.
.PHONY: run
run:
	$(SFTP) $(EXECUTABLE) input.bin goldenOutput.bin root@fp:./
	$(SSH)  root@fp "export OMP_NUM_THREADS=4 && ./filter && rm -rf ./*"

# Cleaning command
RMDIR = rm -rf

clean:
	-$(RMDIR) $(EXECUTABLE)

ECHO := @echo
