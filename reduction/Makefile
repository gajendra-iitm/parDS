# Makefile for compiling reduction.cu

# Compiler and flags
NVCC = nvcc
CXXFLAGS = -O3 -arch=native  # Change sm_35 to your GPU architecture if needed

# Target executable
TARGET = reduction.out

# Source files
SRC = reduction.cu

# Default rule
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Clean rule to remove compiled files
clean:
	rm -f $(TARGET)

