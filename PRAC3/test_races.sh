#!/bin/bash
# Paso 2: validación de corrección con matrices 8x8 y "ruido" de planificación.
# Las versiones correctas deben dar CHECK= SUCCESS; las *_race deben fallar.
make -s all || exit 1
REP=${1:-100000}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

for exe in MMpar_noise MMpar_dep MMpar2_noise MMpar_race MMpar2_race; do
  for dq in 0 4 2; do
    echo "=== ./$exe 8 $dq $REP   (OMP_NUM_THREADS=$OMP_NUM_THREADS)"
    ./$exe 8 $dq $REP | grep -E "CHECK|Result"
  done
done

echo "=== Matrices grandes: el checksum debe coincidir con la versión serie"
./MM     1024 0  | grep Result
./MMpar  1024 0  | grep Result
./MMpar  1024 64 | grep Result
./MMpar2 1024 0  | grep Result
./MMpar2 1024 64 | grep Result
