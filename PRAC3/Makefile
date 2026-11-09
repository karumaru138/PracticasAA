CXX      = g++
CXXFLAGS = -Ofast -march=native -fopenmp -DTYPE=int

ALL = MM MM2 MMpar MMpar2 \
      MMpar_noise MMpar_race MMpar_dep \
      MMpar2_noise MMpar2_race

all: $(ALL)

# Secuenciales (originales)
MM:     MM_main.cpp MM.h     ; $(CXX) $(CXXFLAGS) -include MM.h     $< -o $@
MM2:    MM_main.cpp MM2.h    ; $(CXX) $(CXXFLAGS) -include MM2.h    $< -o $@

# Paralelas
MMpar:  MM_main.cpp MMpar.h  ; $(CXX) $(CXXFLAGS) -include MMpar.h  $< -o $@
MMpar2: MM_main.cpp MMpar2.h ; $(CXX) $(CXXFLAGS) -include MMpar2.h $< -o $@

# Variantes para validar corrección (Paso 2)
MMpar_noise:  MM_main.cpp MMpar.h  ; $(CXX) $(CXXFLAGS) -DNOISE         -include MMpar.h  $< -o $@
MMpar_race:   MM_main.cpp MMpar.h  ; $(CXX) $(CXXFLAGS) -DNOISE -DNOSYNC -include MMpar.h  $< -o $@
MMpar_dep:    MM_main.cpp MMpar.h  ; $(CXX) $(CXXFLAGS) -DNOISE -DDEPEND -include MMpar.h  $< -o $@
MMpar2_noise: MM_main.cpp MMpar2.h ; $(CXX) $(CXXFLAGS) -DNOISE         -include MMpar2.h $< -o $@
MMpar2_race:  MM_main.cpp MMpar2.h ; $(CXX) $(CXXFLAGS) -DNOISE -DNOSYNC -include MMpar2.h $< -o $@

clean: ; rm -f $(ALL)

.PHONY: all clean
