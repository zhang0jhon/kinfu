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
include CMakeFiles/demo1.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo1.dir/flags.make

CMakeFiles/demo1.dir/apps/demo.cpp.o: CMakeFiles/demo1.dir/flags.make
CMakeFiles/demo1.dir/apps/demo.cpp.o: ../apps/demo.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zhang_jhon/kinfu_remake/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/demo1.dir/apps/demo.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/demo1.dir/apps/demo.cpp.o -c /home/zhang_jhon/kinfu_remake/apps/demo.cpp

CMakeFiles/demo1.dir/apps/demo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo1.dir/apps/demo.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zhang_jhon/kinfu_remake/apps/demo.cpp > CMakeFiles/demo1.dir/apps/demo.cpp.i

CMakeFiles/demo1.dir/apps/demo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo1.dir/apps/demo.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zhang_jhon/kinfu_remake/apps/demo.cpp -o CMakeFiles/demo1.dir/apps/demo.cpp.s

CMakeFiles/demo1.dir/apps/demo.cpp.o.requires:
.PHONY : CMakeFiles/demo1.dir/apps/demo.cpp.o.requires

CMakeFiles/demo1.dir/apps/demo.cpp.o.provides: CMakeFiles/demo1.dir/apps/demo.cpp.o.requires
	$(MAKE) -f CMakeFiles/demo1.dir/build.make CMakeFiles/demo1.dir/apps/demo.cpp.o.provides.build
.PHONY : CMakeFiles/demo1.dir/apps/demo.cpp.o.provides

CMakeFiles/demo1.dir/apps/demo.cpp.o.provides.build: CMakeFiles/demo1.dir/apps/demo.cpp.o

# Object files for target demo1
demo1_OBJECTS = \
"CMakeFiles/demo1.dir/apps/demo.cpp.o"

# External object files for target demo1
demo1_EXTERNAL_OBJECTS =

demo1: CMakeFiles/demo1.dir/apps/demo.cpp.o
demo1: CMakeFiles/demo1.dir/build.make
demo1: /usr/lib/x86_64-linux-gnu/libboost_system.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_thread.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
demo1: /usr/lib/x86_64-linux-gnu/libpthread.so
demo1: /usr/local/lib/libpcl_common.so
demo1: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
demo1: /usr/local/lib/libpcl_kdtree.so
demo1: /usr/local/lib/libpcl_octree.so
demo1: /usr/local/lib/libpcl_search.so
demo1: /usr/local/lib/libpcl_sample_consensus.so
demo1: /usr/local/lib/libpcl_filters.so
demo1: /usr/lib/libOpenNI.so
demo1: /usr/lib/libOpenNI2.so
demo1: /usr/lib/libvtkCommon.so.5.8.0
demo1: /usr/lib/libvtkFiltering.so.5.8.0
demo1: /usr/lib/libvtkImaging.so.5.8.0
demo1: /usr/lib/libvtkGraphics.so.5.8.0
demo1: /usr/lib/libvtkGenericFiltering.so.5.8.0
demo1: /usr/lib/libvtkIO.so.5.8.0
demo1: /usr/lib/libvtkRendering.so.5.8.0
demo1: /usr/lib/libvtkVolumeRendering.so.5.8.0
demo1: /usr/lib/libvtkHybrid.so.5.8.0
demo1: /usr/lib/libvtkWidgets.so.5.8.0
demo1: /usr/lib/libvtkParallel.so.5.8.0
demo1: /usr/lib/libvtkInfovis.so.5.8.0
demo1: /usr/lib/libvtkGeovis.so.5.8.0
demo1: /usr/lib/libvtkViews.so.5.8.0
demo1: /usr/lib/libvtkCharts.so.5.8.0
demo1: /usr/local/lib/libpcl_io.so
demo1: /usr/local/lib/libpcl_features.so
demo1: /usr/local/lib/libpcl_keypoints.so
demo1: /usr/lib/x86_64-linux-gnu/libqhull.so
demo1: /usr/local/lib/libpcl_surface.so
demo1: /usr/local/lib/libpcl_visualization.so
demo1: /usr/local/lib/libpcl_registration.so
demo1: /usr/local/lib/libpcl_ml.so
demo1: /usr/local/lib/libpcl_recognition.so
demo1: /usr/local/lib/libpcl_gpu_containers.so
demo1: /usr/local/lib/libpcl_gpu_utils.so
demo1: /usr/local/lib/libpcl_gpu_surface.so
demo1: /usr/local/lib/libpcl_gpu_octree.so
demo1: /usr/local/lib/libpcl_gpu_segmentation.so
demo1: /usr/local/lib/libpcl_gpu_kinfu.so
demo1: /usr/local/lib/libpcl_gpu_features.so
demo1: /usr/local/lib/libpcl_gpu_people.so
demo1: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
demo1: /usr/local/lib/libpcl_segmentation.so
demo1: /usr/local/lib/libpcl_people.so
demo1: /usr/local/lib/libpcl_outofcore.so
demo1: /usr/local/lib/libpcl_tracking.so
demo1: /usr/local/lib/libpcl_stereo.so
demo1: /usr/local/lib/libpcl_apps.so
demo1: /usr/local/lib/libpcl_3d_rec_framework.so
demo1: /usr/local/lib/libpcl_cuda_segmentation.so
demo1: /usr/local/lib/libpcl_cuda_io.so
demo1: /usr/local/lib/libpcl_cuda_features.so
demo1: /usr/local/lib/libpcl_cuda_sample_consensus.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_system.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_thread.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
demo1: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
demo1: /usr/lib/x86_64-linux-gnu/libpthread.so
demo1: /usr/lib/x86_64-linux-gnu/libqhull.so
demo1: /usr/lib/libOpenNI.so
demo1: /usr/lib/libOpenNI2.so
demo1: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
demo1: /usr/lib/libvtkCommon.so.5.8.0
demo1: /usr/lib/libvtkFiltering.so.5.8.0
demo1: /usr/lib/libvtkImaging.so.5.8.0
demo1: /usr/lib/libvtkGraphics.so.5.8.0
demo1: /usr/lib/libvtkGenericFiltering.so.5.8.0
demo1: /usr/lib/libvtkIO.so.5.8.0
demo1: /usr/lib/libvtkRendering.so.5.8.0
demo1: /usr/lib/libvtkVolumeRendering.so.5.8.0
demo1: /usr/lib/libvtkHybrid.so.5.8.0
demo1: /usr/lib/libvtkWidgets.so.5.8.0
demo1: /usr/lib/libvtkParallel.so.5.8.0
demo1: /usr/lib/libvtkInfovis.so.5.8.0
demo1: /usr/lib/libvtkGeovis.so.5.8.0
demo1: /usr/lib/libvtkViews.so.5.8.0
demo1: /usr/lib/libvtkCharts.so.5.8.0
demo1: /usr/local/lib/libpcl_common.so
demo1: /usr/local/lib/libpcl_kdtree.so
demo1: /usr/local/lib/libpcl_octree.so
demo1: /usr/local/lib/libpcl_search.so
demo1: /usr/local/lib/libpcl_sample_consensus.so
demo1: /usr/local/lib/libpcl_filters.so
demo1: /usr/local/lib/libpcl_io.so
demo1: /usr/local/lib/libpcl_features.so
demo1: /usr/local/lib/libpcl_keypoints.so
demo1: /usr/local/lib/libpcl_surface.so
demo1: /usr/local/lib/libpcl_visualization.so
demo1: /usr/local/lib/libpcl_registration.so
demo1: /usr/local/lib/libpcl_ml.so
demo1: /usr/local/lib/libpcl_recognition.so
demo1: /usr/local/lib/libpcl_gpu_containers.so
demo1: /usr/local/lib/libpcl_gpu_utils.so
demo1: /usr/local/lib/libpcl_gpu_surface.so
demo1: /usr/local/lib/libpcl_gpu_octree.so
demo1: /usr/local/lib/libpcl_gpu_segmentation.so
demo1: /usr/local/lib/libpcl_gpu_kinfu.so
demo1: /usr/local/lib/libpcl_gpu_features.so
demo1: /usr/local/lib/libpcl_gpu_people.so
demo1: /usr/local/lib/libpcl_gpu_kinfu_large_scale.so
demo1: /usr/local/lib/libpcl_segmentation.so
demo1: /usr/local/lib/libpcl_people.so
demo1: /usr/local/lib/libpcl_outofcore.so
demo1: /usr/local/lib/libpcl_tracking.so
demo1: /usr/local/lib/libpcl_stereo.so
demo1: /usr/local/lib/libpcl_apps.so
demo1: /usr/local/lib/libpcl_3d_rec_framework.so
demo1: /usr/local/lib/libpcl_cuda_segmentation.so
demo1: /usr/local/lib/libpcl_cuda_io.so
demo1: /usr/local/lib/libpcl_cuda_features.so
demo1: /usr/local/lib/libpcl_cuda_sample_consensus.so
demo1: /usr/lib/libvtkViews.so.5.8.0
demo1: /usr/lib/libvtkInfovis.so.5.8.0
demo1: /usr/lib/libvtkWidgets.so.5.8.0
demo1: /usr/lib/libvtkVolumeRendering.so.5.8.0
demo1: /usr/lib/libvtkHybrid.so.5.8.0
demo1: /usr/lib/libvtkParallel.so.5.8.0
demo1: /usr/lib/libvtkRendering.so.5.8.0
demo1: /usr/lib/libvtkImaging.so.5.8.0
demo1: /usr/lib/libvtkGraphics.so.5.8.0
demo1: /usr/lib/libvtkIO.so.5.8.0
demo1: /usr/lib/libvtkFiltering.so.5.8.0
demo1: /usr/lib/libvtkCommon.so.5.8.0
demo1: /usr/lib/libvtksys.so.5.8.0
demo1: CMakeFiles/demo1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable demo1"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo1.dir/build: demo1
.PHONY : CMakeFiles/demo1.dir/build

CMakeFiles/demo1.dir/requires: CMakeFiles/demo1.dir/apps/demo.cpp.o.requires
.PHONY : CMakeFiles/demo1.dir/requires

CMakeFiles/demo1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo1.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo1.dir/clean

CMakeFiles/demo1.dir/depend:
	cd /home/zhang_jhon/kinfu_remake/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhang_jhon/kinfu_remake /home/zhang_jhon/kinfu_remake /home/zhang_jhon/kinfu_remake/build /home/zhang_jhon/kinfu_remake/build /home/zhang_jhon/kinfu_remake/build/CMakeFiles/demo1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo1.dir/depend

