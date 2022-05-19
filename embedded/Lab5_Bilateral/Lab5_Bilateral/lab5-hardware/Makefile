.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make bin"
	$(ECHO) "      Command to pick the specific files and generates the design for hw target and ARM architecture."
	$(ECHO) ""
	$(ECHO) "  make check TARGET=<sw_emu> HOST_ARCH=<aarch32/aarch64/x86> SYSROOT=<sysroot_path>"
	$(ECHO) "      Command to run application in software emulation."
	$(ECHO) ""
	$(ECHO) "  make exe HOST_ARCH=<aarch32/aarch64/x86>"
	$(ECHO) "      Command to build exe application"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."

# Points to top directory of Git repository
COMMON_REPO = ./lib/
PWD = $(shell readlink -f .)
ABS_COMMON_REPO = $(shell readlink -f $(COMMON_REPO))

# Necessary variables. Those are the default values that you should not change.
TARGET := hw
HOST_ARCH := aarch64
SYSROOT := /home/alex/petalinux2019_2/sysroot/sysroots/aarch64-xilinx-linux/
DEVICE := zcu102_custom_fp
XCLBIN := xclbin
SDCARD := sd_card
FPGA_IP = fp
CONFIG_FILE := design.cfg
KERNEL_NAME := bilateralFilterKernel

# Default values for software emulation on VM x86. Do not change
ifeq ($(TARGET),sw_emu)
	DEVICE := xilinx_u50_xdma_201920_1
	HOST_ARCH := x86
	CONFIG_FILE := design_emu.cfg
endif

# The C++ Compiler to use is included here, depending architexture
include ./lib/utils.mk

# The kernel Compiler to use : V++
VPP := v++

# Include Libraries for OpenCL and Xilinx Runtime
# and various utility functions -  DO NOT CHANGE
include $(ABS_COMMON_REPO)/common/includes/opencl/opencl.mk
include $(ABS_COMMON_REPO)/common/includes/xcl2/xcl2.mk
XSA := $(call device2xsa, $(DEVICE))
BUILD_DIR := ./build/build_dir.$(TARGET).$(XSA)
BUILD_DIR_hwKernels = $(BUILD_DIR)/$(KERNEL_NAME)


# ####### Host ###########

# The below are compile flags are passed to the C++ Compiler
CXXFLAGS += -lm -Wall -O3 -g -fopenmp
CXXFLAGS += $(xcl2_CXXFLAGS) $(opencl_CXXFLAGS)

# The below are linking flags for C++ Compiler
LDFLAGS += $(opencl_LDFLAGS) $(xcl2_LDFLAGS)
ifneq ($(HOST_ARCH), x86)
	LDFLAGS += --sysroot=$(SYSROOT)
endif

HOST_C_SRCS += filterHost.c
EXECUTABLE = filter


########## HLS ###########

# Kernel compiler global settings
CLFLAGS += -t $(TARGET) --jobs 16 --platform $(DEVICE) --config $(CONFIG_FILE) --save-temps
ifneq ($(TARGET), hw)
	CLFLAGS += -g
endif
CLFLAGS +=  --advanced.prop kernel.bilateralFilterKernel.kernel_flags="-lm"

#HLS C++ Files
HLS_CPP_SRCS += filterHLS.cpp
# HLS Object Files
BINARY_CONTAINERS += $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
BINARY_CONTAINER_bilateralFilterKernel_OBJS += $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xo

########## Utilities ##########
CP = cp -rf
SFTP = scp
SSH = ssh
XCLBINITUTIL = xclbinutil

########## Rules ##########
# Building Host
# It simply calls the appropriate Compiler (x86 gcc or ARM gcc) to build the executable
.PHONY: exe
exe: $(EXECUTABLE)

$(EXECUTABLE): $(HOST_C_SRCS)
	$(CXX) $(CXXFLAGS) $(HOST_C_SRCS) -o '$@' $(LDFLAGS)
	mkdir -p $(XCLBIN)
	$(CP) $(EXECUTABLE) $(XCLBIN)


# Building hw kernel
# Uses Vitis compiler with the appropriate flags to generate .xclbin file
.PHONY: bin
bin: $(BINARY_CONTAINERS)

$(BINARY_CONTAINER_bilateralFilterKernel_OBJS): $(HLS_CPP_SRCS)
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR_hwKernels) -c -k $(KERNEL_NAME) -I'$(<D)' -o'$@' '$<'
$(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin: $(BINARY_CONTAINER_bilateralFilterKernel_OBJS)
	mkdir -p $(XCLBIN)
	$(VPP) $(CLFLAGS) --temp_dir $(BUILD_DIR_hwKernels) -l $(LDCLFLAGS) -o'$@' $(+)
ifneq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
	mkdir -p reports
	$(CP) $(BUILD_DIR_hwKernels)/reports/* ./reports/
	$(CP) $(BUILD_DIR_hwKernels)/reports/link/imp/kernel_util_synthed.rpt ./

endif

# Rules for software emulation
EMCONFIG_DIR = $(XCLBIN)
EMU_DIR = $(SDCARD)/data/emulation

.PHONY: all clean cleanall emconfig
all: $(EXECUTABLE) $(BINARY_CONTAINERS) Makefile

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

check: all emconfig
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	export XCL_EMULATION_MODE=sw_emu && ./$(EXECUTABLE) $(XCLBIN)/$(KERNEL_NAME).$(TARGET).xclbin
endif
else
ifeq ($(HOST_ARCH), x86)
	 ./$(EXECUTABLE)
endif
endif

# Run rule.
# It creates a script (run.sh) that contains the necessary commands to execute the design on FPGA.
# Then it uploads it on FPGA, and starts an ssh sessions that executes that script.
# After that, it cleans up the FPGA by deleting the uploaded files. The results are shown on console.
.PHONY: run
run:
	$(ECHO) 'chmod a+x $(EXECUTABLE)' > run.sh
	$(ECHO) 'fpgautil -b $(KERNEL_NAME).$(TARGET).bit' >> run.sh
	$(ECHO) './$(EXECUTABLE) $(KERNEL_NAME).$(TARGET).xclbin' >> run.sh
	$(XCLBINITUTIL) --force --dump-section BITSTREAM:RAW:$(KERNEL_NAME).$(TARGET).bit -i $(BINARY_CONTAINERS)
	$(SFTP) ./$(XCLBIN)/$(EXECUTABLE) root@$(FPGA_IP):./
	$(SFTP) $(KERNEL_NAME).$(TARGET).bit root@$(FPGA_IP):./
	$(SFTP) $(BINARY_CONTAINERS) root@$(FPGA_IP):./
	$(SFTP) run.sh input.bin goldenOutput.bin root@$(FPGA_IP):./
	$(SSH)  root@$(FPGA_IP) "export XILINX_XRT=/usr && chmod a+x run.sh && ./run.sh && rm -rf ./*"

# Cleaning stuff
RMDIR = rm -rf

clean:
	-$(RMDIR) $(EXECUTABLE)
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv
	-$(RMDIR) host_src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) run.sh

cleanall: clean
	-$(RMDIR) $(XCLBIN)
	-$(RMDIR) _x.* *xclbin.run_summary qemu-memory-_* emulation/ _vimage/ pl* start_simulation.sh *.xclbin


ECHO := @echo

######## Not usable #########
# Enable Profiling
REPORT := no # DO NOT CHANGE
PROFILE:= no # DO NOT CHANGE
#timeline_trace is always enabled
#'estimate' for estimate report generation
#'system' for system report generation
ifneq ($(REPORT), no)
CLFLAGS += --report estimate
CLLDFLAGS += --report system
endif

#Generates profile summary report
ifeq ($(PROFILE), yes)
LDCLFLAGS += --profile_kernel data:all:all:all:all
LDCFLAGS += --profile_kernel  stall:all:all:all:all
LDCFALGS += --profile_kernel exec:all:all:all:all
endif
