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

# Utility rule file for pagerank-benchmark.

# Include any custom commands dependencies for this target.
include bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/compiler_depend.make

# Include the progress variables for this target.
include bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/progress.make

bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark: bin/nvm-pagerank-bench

pagerank-benchmark: bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark
pagerank-benchmark: bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/build.make
.PHONY : pagerank-benchmark

# Rule to build all files generated by this target.
bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/build: pagerank-benchmark
.PHONY : bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/build

bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/clean:
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/pagerank && $(CMAKE_COMMAND) -P CMakeFiles/pagerank-benchmark.dir/cmake_clean.cmake
.PHONY : bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/clean

bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/depend:
	cd /home/ErHa/GPUANNsIndex_GDS11.28/anns/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ErHa/GPUANNsIndex_GDS11.28/anns /home/ErHa/GPUANNsIndex_GDS11.28/anns/bam/benchmarks/pagerank /home/ErHa/GPUANNsIndex_GDS11.28/anns/build /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/pagerank /home/ErHa/GPUANNsIndex_GDS11.28/anns/build/bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : bam/benchmarks/pagerank/CMakeFiles/pagerank-benchmark.dir/depend

