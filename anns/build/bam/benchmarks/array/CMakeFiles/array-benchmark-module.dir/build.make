# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ErHa/GPUANNsIndex_GDS11.28/anns

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ErHa/GPUANNsIndex_GDS11.28/anns/build

# Include any dependencies generated for this target.
include bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/compiler_depend.make

# Include the progress variables for this target.
include bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/progress.make

# Include the compile flags for this target's objects.
include bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/flags.make

bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o: /home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/benchmarks/array/main.cu
bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o: bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o.depend
bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o: bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o.cmake
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o"
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir && /usr/local/bin/cmake -E make_directory /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir//.
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir && /usr/local/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir//./array-benchmark-module_generated_main.cu.o -D generated_cubin_file:STRING=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir//./array-benchmark-module_generated_main.cu.o.cubin.txt -P /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir//array-benchmark-module_generated_main.cu.o.cmake

# Object files for target array-benchmark-module
array__benchmark__module_OBJECTS =

# External object files for target array-benchmark-module
array__benchmark__module_EXTERNAL_OBJECTS = \
"/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o"

bin/nvm-array-bench: bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o
bin/nvm-array-bench: bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/build.make
bin/nvm-array-bench: /usr/local/cuda-12.3/lib64/libcudart_static.a
bin/nvm-array-bench: /usr/lib/x86_64-linux-gnu/librt.a
bin/nvm-array-bench: lib/libnvm.so
bin/nvm-array-bench: /usr/local/cuda-12.3/lib64/libcudart_static.a
bin/nvm-array-bench: /usr/lib/x86_64-linux-gnu/librt.a
bin/nvm-array-bench: /usr/local/cuda-12.3/lib64/libcudart_static.a
bin/nvm-array-bench: /usr/lib/x86_64-linux-gnu/librt.a
bin/nvm-array-bench: bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/ErHa/GPUANNsIndex_GDS11.28/anns/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../../../bin/nvm-array-bench"
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/array-benchmark-module.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/build: bin/nvm-array-bench
.PHONY : bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/build

bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/clean:
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array && $(CMAKE_COMMAND) -P CMakeFiles/array-benchmark-module.dir/cmake_clean.cmake
.PHONY : bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/clean

bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/depend: bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/array-benchmark-module_generated_main.cu.o
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ErHa/GPUANNsIndex_GDS11.28/anns /home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/benchmarks/array /home/ErHa/GPUANNsIndex_GDS11.28/anns/build /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : bam/benchmarks/array/CMakeFiles/array-benchmark-module.dir/depend

