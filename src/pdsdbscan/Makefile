EXECUTABLE := main
CC_FILES   := *.cpp


all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -std=c++14 -m64
CXXFLAGS=-O3 -Wall

OBJS=$(OBJDIR)/main.o  $(OBJDIR)/pdsdbscan.o

.PHONY: dirs clean

all: $(EXECUTABLE)

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)


$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

# $(OBJDIR)/main.o: main.cpp dbscan.h
# 	g++ $(CPPFLAGS) -c main.cpp

# $(OBJDIR)/dbscan.o: dbscan.cpp dbscan.h
# 	g++ $(CPPFLAGS) -c dbscan.cpp