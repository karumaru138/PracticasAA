#!/bin/bash
# Pasos 3 y 4: análisis de rendimiento. Uso: ./bench.sh [N]   (por defecto 2048)
make -s all || exit 1
N=${1:-2048}
NCPU=$(nproc)
PERF="perf stat -e task-clock,context-switches,cpu-migrations"
t() { # imprime: tiempo transcurrido y CPUs utilizadas
  $PERF "$@" 2>&1 >/dev/null | awk '/CPUs utilized/{cpu=$5} /seconds time elapsed/{tm=$1} END{printf "%8.3f s  %5s CPUs\n", tm, cpu}'
}

echo "### 1) Línea base secuencial  N=$N"
printf "MM   iterativa      : "; t ./MM  $N 0
printf "MM   D&C DQSZ=64    : "; t ./MM  $N 64

echo "### 2) Efecto de DQSZ  (k = $NCPU hilos)"
export OMP_NUM_THREADS=$NCPU
for dq in 4 8 16 32 64 128 256 512 1024; do
  printf "MMpar  DQSZ=%-5s : " $dq; t ./MMpar $N $dq
done
printf "MMpar  iterativa   : "; t ./MMpar $N 0

echo "### 3) Efecto de k  (DQSZ fijo = ${DQ:=64})"
for k in 1 2 4 6 8 12 16 24; do
  [ $k -gt $((2*NCPU)) ] && break
  export OMP_NUM_THREADS=$k
  printf "k=%-3s D&C: " $k; t ./MMpar $N $DQ
  printf "k=%-3s ITE: " $k; t ./MMpar $N 0
done

echo "### 4) Optimizaciones (MM2 / MMpar2)  k=$NCPU"
export OMP_NUM_THREADS=$NCPU
printf "MM2    iterativa serie : "; t ./MM2    $N 0
printf "MMpar2 iterativa       : "; t ./MMpar2 $N 0
for dq in 32 64 128 256 512; do
  printf "MMpar2 DQSZ=%-5s      : " $dq; t ./MMpar2 $N $dq
done
