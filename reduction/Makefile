# Makefile for compiling reduction.cu

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -arch=native

# Target executable
TARGET = reduction.out

# Source files
SRC = reduction.cu

# Default rule
all: $(TARGET)

omp:
	

# Rule to build the target executable
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)

