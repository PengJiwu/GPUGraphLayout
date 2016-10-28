CC:=gcc

# Directories
L_SRC_DIR         := ../../lib
SRC_DIR           := ../../src
OBJ_DIR           := obj

# C/C++ compiler flags
COMMON_FLAGS:=-O3
CFLAGS   :=$(COMMON_FLAGS) -D NO_FREETYPE
CXXFLAGS :=$(COMMON_FLAGS) -std=c++11 #-g
NVCCFLAGS:=#-G

# C preprocessor flags
CPPFLAGS :=-D NO_FREETYPE -isystem $(L_SRC_DIR)/libpng-1.6.23/
ifeq ($(CUDA_SUPPORT), 1)
	CPPFLAGS:=$(CPPFLAGS) -D __CUDA__ -isystem /usr/local/cuda/include/
endif
ifeq ($(DEBUG_PRINTING), 1)
	CPPFLAGS:=$(CPPFLAGS) -D DEBUGPRINT
endif

# Linker flags
LDFLAGS  :=-lc -lm -lstdc++
ifeq ($(CUDA_SUPPORT), 1)
	LDFLAGS:=$(LDFLAGS) -L/usr/local/cuda/lib64 -lcudart
endif

# src./obj. files


GRAPH_VIEWER_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
GRAPH_VIEWER_OBJS := $(GRAPH_VIEWER_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

SNAP_SRCS         := $(L_SRC_DIR)/snap/snap-core/Snap.cpp
SNAP_OBJS         := $(SNAP_SRCS:$(L_SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

PNGWRITER_SRCS    := $(L_SRC_DIR)/pngwriter/src/pngwriter.cc
PNGWRITER_OBJS    := $(PNGWRITER_SRCS:$(L_SRC_DIR)/%.cc=$(OBJ_DIR)/%.o)

LIBPNG_SRCS       := $(wildcard $(L_SRC_DIR)/libpng-1.6.23/*.c)
LIBPNG_SRCS       := $(filter-out $(L_SRC_DIR)/libpng-1.6.23/example.c, $(LIBPNG_SRCS))
LIBPNG_SRCS       := $(filter-out $(L_SRC_DIR)/libpng-1.6.23/pngtest.c, $(LIBPNG_SRCS))
LIBPNG_OBJS       := $(LIBPNG_SRCS:$(L_SRC_DIR)/%.c=$(OBJ_DIR)/%.o)

ZLIB_SRCS         := $(wildcard $(L_SRC_DIR)/zlib-1.2.8/*.c)
ZLIB_OBJS         := $(ZLIB_SRCS:$(L_SRC_DIR)%.c=$(OBJ_DIR)%.o)

CUDA_SRCS         := $(wildcard $(SRC_DIR)/*.cu)
CUDA_OBJS         := $(CUDA_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
CUDA_DEPS         := $(wildcard $(SRC_DIR)/*.cuh)

# Don't compile with CUDA support by default
CUDA_SUPPORT      ?= 0

C_SRC             := $(LIBPNG_SRCS) $(ZLIB_SRCS)
CPP_SRC           := $(GRAPH_VIEWER_SRCS) $(SNAP_SRCS) $(PNGWRITER_SRCS)
CUDA_SRC          := $(CUDA_SRCS)
SOURCES           := $(C_SRC) $(CPP_SRC)
OBJECTS           := $(GRAPH_VIEWER_OBJS) $(SNAP_OBJS) $(PNGWRITER_OBJS) $(LIBPNG_OBJS) $(ZLIB_OBJS)

ifeq ($(CUDA_SUPPORT), 1)
OBJECTS           := $(OBJECTS) $(CUDA_OBJS)
SOURCES           := $(SOURCES) $(CUDA_SRCS)
endif

# Generate dependency (.h, .hpp)
# ala http://stackoverflow.com/questions/2394609/makefile-header-dependencies
depend: .depend
.depend: $(SRCS)
	rm -f ./.depend
	$(CC) $(CFLAGS)   $(CPPFLAGS) -MM -isystem $(L_SRC_DIR)/libpng-1.6.23/ $(C_SRC) >> ./.depend
	$(CC) $(CXXFLAGS) $(CPPFLAGS) -MM -isystem $(L_SRC_DIR)/libpng-1.6.23/ -I $(L_SRC_DIR)/snap/glib-core $(CPP_SRC) >> ./.depend

include ./.depend

graph_viewer: $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o graph_viewer

$(GRAPH_VIEWER_OBJS): $(GRAPH_VIEWER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -I $(L_SRC_DIR)/snap/glib-core -isystem $(L_SRC_DIR)/libpng-1.6.23/ --std=c++11 -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cpp)

$(CUDA_OBJS): $(CUDA_SRCS) $(CUDA_DEPS)
	mkdir -p $(@D)
	nvcc -c $(CXXFLAGS) $(NVCCFLAGS) $(CPPFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(SRC_DIR)/%.cu)

$(SNAP_OBJS): $(SNAP_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -I $(L_SRC_DIR)/snap/glib-core/ -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.cpp)

$(PNGWRITER_OBJS): $(PNGWRITER_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CXXFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.cc)

$(LIBPNG_OBJS): $(LIBPNG_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.c)

$(ZLIB_OBJS): $(ZLIB_SRCS)
	mkdir -p $(@D)
	$(CC) -c $(CPPFLAGS) $(CFLAGS) -o $@ $(@:$(OBJ_DIR)/%.o=$(L_SRC_DIR)/%.c)

clear: clean

clean: 
	rm -r graph_viewer $(OBJ_DIR)/* ./.depend

.PHONY: all clear clean depend
