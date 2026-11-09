// =====================================================================
//  MMpar.h  -  Versiones PARALELAS (OpenMP) de la multiplicación de matrices
//              basadas en MM.h (bucles i-j-k originales, sin optimizar)
//
//  Compilación:
//    g++ -Ofast -march=native -fopenmp -DTYPE=int -include MMpar.h MM_main.cpp -o MMpar
//
//  Macros opcionales para experimentos de corrección (Paso 2):
//    -DNOISE   : introduce "ruido" de planificación (sched_yield) en el caso
//                base, para provocar entrelazados distintos entre hilos.
//    -DNOSYNC  : introduce A PROPÓSITO errores de concurrencia:
//                  * Iterativa: j y k quedan compartidas entre hilos.
//                  * D&C      : se elimina el taskwait entre las dos fases,
//                               por lo que las tareas i e i+4 escriben a la
//                               vez el mismo cuadrante de c.
//    -DDEPEND  : versión D&C alternativa sincronizada con cláusulas depend
//                (en lugar de taskwait entre fases).
// =====================================================================
#include <omp.h>
#ifdef NOISE
#include <sched.h>
#define MAKE_NOISE() sched_yield()
#else
#define MAKE_NOISE()
#endif

// ---------------------------------------------------------------------
//  Versión iterativa: paralelismo de bucle
//  Cada hilo recibe un bloque contiguo de filas i (schedule static):
//    - escribe sólo sus filas de c  -> no hay escrituras compartidas
//    - lee sus filas de a y TODA la matriz b (sólo lectura)
// ---------------------------------------------------------------------
template <class type>
  void MM ( const type *a, const type *b, type *c, const int N )
{
#ifndef NOSYNC
  #pragma omp parallel for schedule(static)
  for (int i=0; i<N; i++)
    for (int j=0; j<N; j++)
    {
      MAKE_NOISE();
      for (int k=0; k<N; k++)
        c[i*N+j] += a[i*N+k]*b[k*N+j];
    }
#else
  // ERROR INTENCIONADO: j y k se declaran fuera de la región paralela y
  // no son private -> los hilos comparten los índices (data race).
  int i,j,k;
  #pragma omp parallel for schedule(static)
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
    {
      MAKE_NOISE();
      for (k=0; k<N; k++)
        c[i*N+j] += a[i*N+k]*b[k*N+j];
    }
#endif
}

// ---------------------------------------------------------------------
//  Versión Divide y Vencerás: paralelismo de tareas
// ---------------------------------------------------------------------
static int DQSZ; // Smaller Size for a subproblem

// Caso base: algoritmo clásico sobre un bloque SZ x SZ
template <class type>
  static inline void MM_base_block ( const type *a, const type *b, type *c,
                                     const int SZ, const int N )
{
  MAKE_NOISE();
  for (int i=0; i<SZ; i++)
    for (int j=0; j<SZ; j++)
      for (int k=0; k<SZ; k++)
        c[i*N+j] += a[i*N+k]*b[k*N+j];
}

// c[][] = c[][] + a[][] * b[][]   (se ejecuta DENTRO de la región paralela)
//
// Análisis de dependencias de las 8 llamadas:
//   Fase 1: C11+=A11*B11  C12+=A11*B12  C21+=A21*B11  C22+=A21*B12
//   Fase 2: C11+=A12*B21  C12+=A12*B22  C21+=A22*B21  C22+=A22*B22
//   Dentro de cada fase las 4 tareas escriben cuadrantes DISJUNTOS de c
//   (a y b sólo se leen) -> pueden ejecutarse en paralelo.
//   La tarea t de la fase 2 escribe el mismo cuadrante que la tarea t de la
//   fase 1 -> dependencia de salida (WAW + RAW sobre c): hay que esperar.
template <class type>
  void MM_DQ_task ( const type *a, const type *b, type *c, int SZ, const int N )
{
  if (SZ <= DQSZ)
  {
    MM_base_block<type>(a, b, c, SZ, N);
    return;
  }

  SZ = SZ/2;  // assume SZ is a perfect power of 2

#if defined(DEPEND)
  // Alternativa: dependencias explícitas sobre el primer elemento de cada
  // cuadrante de c (identifica de forma única el cuadrante).
  type *c11 = c, *c12 = c+SZ, *c21 = c+SZ*N, *c22 = c+SZ*(N+1);
  #pragma omp task depend(inout: c11[0])
  MM_DQ_task<type> ( a,          b,          c11, SZ, N);
  #pragma omp task depend(inout: c12[0])
  MM_DQ_task<type> ( a,          b+SZ,       c12, SZ, N);
  #pragma omp task depend(inout: c21[0])
  MM_DQ_task<type> ( a+SZ*N,     b,          c21, SZ, N);
  #pragma omp task depend(inout: c22[0])
  MM_DQ_task<type> ( a+SZ*N,     b+SZ,       c22, SZ, N);

  #pragma omp task depend(inout: c11[0])
  MM_DQ_task<type> ( a+SZ,       b+SZ*N,     c11, SZ, N);
  #pragma omp task depend(inout: c12[0])
  MM_DQ_task<type> ( a+SZ,       b+SZ*(N+1), c12, SZ, N);
  #pragma omp task depend(inout: c21[0])
  MM_DQ_task<type> ( a+SZ*(N+1), b+SZ*N,     c21, SZ, N);
  #pragma omp task depend(inout: c22[0])
  MM_DQ_task<type> ( a+SZ*(N+1), b+SZ*(N+1), c22, SZ, N);
  #pragma omp taskwait
#else
  // ---- Fase 1: 4 tareas independientes ----
  #pragma omp task
  MM_DQ_task<type> ( a,          b,          c,          SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a,          b+SZ,       c+SZ,       SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*N,     b,          c+SZ*N,     SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*N,     b+SZ,       c+SZ*(N+1), SZ, N);

#ifndef NOSYNC
  #pragma omp taskwait   // la fase 2 vuelve a escribir los mismos cuadrantes
#endif

  // ---- Fase 2: 4 tareas independientes ----
  #pragma omp task
  MM_DQ_task<type> ( a+SZ,       b+SZ*N,     c,          SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ,       b+SZ*(N+1), c+SZ,       SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*(N+1), b+SZ*N,     c+SZ*N,     SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*(N+1), b+SZ*(N+1), c+SZ*(N+1), SZ, N);

  // Imprescindible: el padre no puede terminar (y su propio padre no puede
  // empezar la fase 2) hasta que todo su sub-árbol haya acabado.
  #pragma omp taskwait
#endif
}

// Punto de entrada llamado desde MM_main.cpp (misma firma que en MM.h):
// crea el equipo de hilos y un único hilo genera el árbol de tareas.
template <class type>
  void MM_DQ ( const type *a, const type *b, type *c, int SZ, const int N )
{
  #pragma omp parallel
  #pragma omp single
  MM_DQ_task<type>(a, b, c, SZ, N);
}
