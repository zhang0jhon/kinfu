# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhang_jhon/kinfu_remake

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhang_jhon/kinfu_remake/build

# Include any dependencies generated for this target.
include apps/CMakeFiles/demo.dir/depend.make

# Include the progress variables for this target.
include apps/CMakeFiles/demo.dir/progress.make

# Include the compile flags for this target's objects.
include apps/CMakeFiles/demo.dir/flags.make

apps/CMakeFiles/demo.dir/demo.cpp.o: apps/CMakeFiles/demo.dir/flags.make
apps/CMakeFiles/demo.dir/demo.cpp.o: ../apps/demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zhang_jhon/kinfu_remake/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object apps/CMakeFiles/demo.dir/demo.cpp.o"
	cd /home/zhang_jhon/kinfu_remake/build/apps && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/demo.dir/demo.cpp.o -c /home/zhang_jhon/kinfu_remake/apps/demo.cpp

apps/CMakeFiles/demo.dir/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo.dir/demo.cpp.i"
	cd /home/zhang_jhon/kinfu_remake/build/apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zhang_jhon/kinfu_remake/apps/demo.cpp > CMakeFiles/demo.dir/demo.cpp.i

apps/CMakeFiles/demo.dir/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo.dir/demo.cpp.s"
	cd /home/zhang_jhon/kinfu_remake/build/apps && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zhang_jhon/kinfu_remake/apps/demo.cpp -o CMakeFiles/demo.dir/demo.cpp.s

apps/CMakeFiles/demo.dir/demo.cpp.o.requires:
.PHONY : apps/CMakeFiles/demo.dir/demo.cpp.o.requires

apps/CMakeFiles/demo.dir/demo.cpp.o.provides: apps/CMakeFiles/demo.dir/demo.cpp.o.requires
	$(MAKE) -f apps/CMakeFiles/demo.dir/build.make apps/CMakeFiles/demo.dir/demo.cpp.o.provides.build
.PHONY : apps/CMakeFiles/demo.dir/demo.cpp.o.provides

apps/CMakeFiles/demo.dir/demo.cpp.o.provides.build: apps/CMakeFiles/demo.dir/demo.cpp.o

# Object files for target demo
demo_OBJECTS = \
"CMakeFiles/demo.dir/demo.cpp.o"

# External object files for target demo
demo_EXTERNAL_OBJECTS =

bin/demo: apps/CMakeFiles/demo.dir/demo.cpp.o
bin/demo: apps/CMakeFiles/demo.dir/build.make
bin/demo: /usr/local/lib/libopencv_core.so.3.0.0
bin/demo: /usr/local/lib/libopencv_viz.so.3.0.0
bin/demo: /usr/local/lib/libopencv_highgui.so.3.0.0
bin/demo: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
bin/demo: lib/libkfusion.a
bin/demo: /usr/local/lib/libopencv_viz.so.3.0.0
bin/demo: /usr/local/lib/libopencv_highgui.so.3.0.0
bin/demo: /usr/local/lib/libopencv_imgcodecs.so.3.0.0
bin/demo: /usr/local/lib/libopencv_core.so.3.0.0
bin/demo: /usr/local/lib/libopencv_cudev.so.3.0.0
bin/demo: /usr/local/lib/libopencv_hal.a
bin/demo: /usr/local/share/OpenCV/3rdparty/lib/libippicv.a
bin/demo: /usr/local/cuda-7.5/lib64/libcudart.so
bin/demo: /usr/lib/x86_64-linux-gnu/libcuda.so
bin/demo: /usr/lib/libOpenNI.so
bin/demo: apps/CMakeFiles/demo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/demo"
	cd /home/zhang_jhon/kinfu_remake/build/apps && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
apps/CMakeFiles/demo.dir/build: bin/demo
.PHONY : apps/CMakeFiles/demo.dir/build

apps/CMakeFiles/demo.dir/requires: apps/CMakeFiles/demo.dir/demo.cpp.o.requires
.PHONY : apps/CMakeFiles/demo.dir/requires

apps/CMakeFiles/demo.dir/clean:
	cd /home/zhang_jhon/kinfu_remake/build/apps && $(CMAKE_COMMAND) -P CMakeFiles/demo.dir/cmake_clean.cmake
.PHONY : apps/CMakeFiles/demo.dir/clean

apps/CMakeFiles/demo.dir/depend:
	cd /home/zhang_jhon/kinfu_remake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhang_jhon/kinfu_remake /home/zhang_jhon/kinfu_remake/apps /home/zhang_jhon/kinfu_remake/build /home/zhang_jhon/kinfu_remake/build/apps /home/zhang_jhon/kinfu_remake/build/apps/CMakeFiles/demo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : apps/CMakeFiles/demo.dir/depend

