// =====================================================================
//  MMpar2.h - Versiones PARALELAS (OpenMP) + optimizaciones de MM2.h
//     Optimización 1: intercambio de bucles (i-k-j): el bucle interno
//                     recorre b y c por filas (acceso secuencial, stride 1)
//                     -> localidad espacial + vectorización SIMD.
//     Optimización 2: register blocking / desenrollado 2x2 de los dos bucles
//                     externos: cada elemento de b cargado se usa para 2
//                     filas de c y cada c[] cargado acumula 2 productos
//                     -> la mitad de accesos a memoria por operación.
//
//  Compilación:
//    g++ -Ofast -march=native -fopenmp -DTYPE=int -include MMpar2.h MM_main.cpp -o MMpar2
//
//  Requisitos: N potencia de 2 y DQSZ >= 2 (el desenrollado es de 2x2).
//  Macros opcionales: -DNOISE, -DNOSYNC, -DDEPEND (ver MMpar.h)
// =====================================================================
#include <omp.h>
#ifdef NOISE
#include <sched.h>
#define MAKE_NOISE() sched_yield()
#else
#define MAKE_NOISE()
#endif

// Núcleo optimizado: c[ROWS x SZ] += a[ROWS x SZ] * b[SZ x SZ]
// (las filas tienen longitud N). En D&C ROWS = SZ (bloque cuadrado).
// __restrict: c no se solapa con a ni b -> el compilador puede vectorizar
// el bucle j sin comprobaciones de aliasing.
template <class type>
  static inline void MM2_kernel ( const type * __restrict a,
                                  const type * __restrict b,
                                  type * __restrict c,
                                  const int ROWS, const int SZ, const int N )
{
  for (int i=0; i<ROWS; i+=2)
    for (int k=0; k<SZ; k+=2)
    {
      const type A00 = a[  i  *N+k], A01 = a[  i  *N+k+1];
      const type A10 = a[(i+1)*N+k], A11 = a[(i+1)*N+k+1];
      const type *b0 = b +   k  *N;
      const type *b1 = b + (k+1)*N;
      type *c0 = c +   i  *N;
      type *c1 = c + (i+1)*N;
      #pragma omp simd
      for (int j=0; j<SZ; j++)
      {
        const type B1 = b0[j];
        const type B2 = b1[j];
        c0[j] += A00*B1 + A01*B2;
        c1[j] += A10*B1 + A11*B2;
      }
    }
}

// ---------------------------------------------------------------------
//  Versión iterativa (paralelismo de bucle sobre pares de filas)
// ---------------------------------------------------------------------
template <class type>
  void MM ( const type *a, const type *b, type *c, const int N )
{
#ifndef NOSYNC
  #pragma omp parallel for schedule(static)
  for (int i=0; i<N; i+=2)
  {
    MAKE_NOISE();
    MM2_kernel<type>(a + i*N, b, c + i*N, 2, N, N);   // 2 filas, todas las columnas
  }
#else
  // ERROR INTENCIONADO: se reparte el bucle k entre hilos -> hilos distintos
  // acumulan sobre el MISMO elemento c[i][j] sin sincronización.
  for (int i=0; i<N; i+=2)
  {
    #pragma omp parallel for schedule(static)
    for (int k=0; k<N; k+=2)
    {
      MAKE_NOISE();
      for (int j=0; j<N; j++)
      {
        type B1 = b[k*N+j];
        type B2 = b[(k+1)*N+j];
        c[  i  *N+j] += a[  i  *N+k]*B1 + a[  i  *N+k+1]*B2;
        c[(i+1)*N+j] += a[(i+1)*N+k]*B1 + a[(i+1)*N+k+1]*B2;
      }
    }
  }
#endif
}

// ---------------------------------------------------------------------
//  Versión Divide y Vencerás (paralelismo de tareas)
// ---------------------------------------------------------------------
static int DQSZ; // Smaller Size for a subproblem

template <class type>
  void MM_DQ_task ( const type *a, const type *b, type *c, int SZ, const int N )
{
  if (SZ <= DQSZ)
  {
    MAKE_NOISE();
    MM2_kernel<type>(a, b, c, SZ, SZ, N);
    return;
  }

  SZ = SZ/2;  // assume SZ is a perfect power of 2

#if defined(DEPEND)
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
  #pragma omp task
  MM_DQ_task<type> ( a,          b,          c,          SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a,          b+SZ,       c+SZ,       SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*N,     b,          c+SZ*N,     SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*N,     b+SZ,       c+SZ*(N+1), SZ, N);

#ifndef NOSYNC
  #pragma omp taskwait
#endif

  #pragma omp task
  MM_DQ_task<type> ( a+SZ,       b+SZ*N,     c,          SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ,       b+SZ*(N+1), c+SZ,       SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*(N+1), b+SZ*N,     c+SZ*N,     SZ, N);
  #pragma omp task
  MM_DQ_task<type> ( a+SZ*(N+1), b+SZ*(N+1), c+SZ*(N+1), SZ, N);
  #pragma omp taskwait
#endif
}

template <class type>
  void MM_DQ ( const type *a, const type *b, type *c, int SZ, const int N )
{
  #pragma omp parallel
  #pragma omp single
  MM_DQ_task<type>(a, b, c, SZ, N);
}
