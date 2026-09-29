CXX = nvcc
CXXFLAGS = -O3 -arch=sm_75
LIBS = -lcublas
TARGET = llama_engine
SRCS = main.cu kernels.cu
HEADERS = config.h kernels.h

$(TARGET): $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) prompt.bin output.bin