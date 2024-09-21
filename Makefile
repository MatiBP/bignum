# Compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -g -O2

# Source files
SRCS = Bignum.cpp

HEADERS = Bignum.hpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable
TARGET = bignum

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
