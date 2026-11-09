#include <iostream>
#include <cstring>

// If TYPE is no defined as a compiler option, it is defined here
#ifndef TYPE
#define TYPE int
#endif

// Classical Matrix Multiply
template <class type>
  void MM_base ( const type *a, const type *b, type *c, const int N )
{
  int i,j,k;
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      for (k=0; k<N; k++)
        c[i*N+j] += a[i*N+k]*b[k*N+j];
}

template <class type>
  int compare( type *M1, type *M2, const int N )
{
  int Different=0, j, k;

  for (j=0; j<N; j++)
    for (k=0; k<N; k++) 
      Different = Different | ( M1[j*N+k] != M2[j*N+k]);

  return Different;
}

template <class type>
  type add_elements( type *M, const int N )
{
  type R=0.0;
  int j,k;

  for (j=0; j<N; j++)
    for (k=0; k<N; k++) 
      R += M[j*N+k];

  return R;
}

template <class type>
  void init_mat ( type *M, int N, unsigned int seed)
{
  int k,j;
  srandom(seed);

  for (k=0; k<N; k++) 
    for (j=0; j<N; j++)
      M[k*N+j] = random();
}

int main (int argc, char **argv)
{
  int N, check=0;

  if (argc<3) {
     std::cout << "args: N (Matrix Size) SZ (Block SIZE for Base Case in Divide&Conquer)\n";
     return 0;
  }

  N   = atol(argv[1]);
  DQSZ= atol(argv[2]);
  if (argc > 3) check= atol(argv[3]);

  // allocate memory aligned to 64-Byte boundaries
  TYPE *A  = (TYPE *) aligned_alloc ( 64, N*N*sizeof(TYPE));
  TYPE *B  = (TYPE *) aligned_alloc ( 64, N*N*sizeof(TYPE));
  TYPE *C  = (TYPE *) aligned_alloc ( 64, N*N*sizeof(TYPE));
  TYPE *C2 = (TYPE *) aligned_alloc ( 64, N*N*sizeof(TYPE) );

  init_mat<TYPE>(A, N, 1); init_mat<TYPE>(B, N, 2);

  std::cout << "Matrix Multiply " << N << " x " << N;
  if (DQSZ)
    std::cout << " Recursive up to " << DQSZ << " x " << DQSZ << " blocks\n";
  else
    std::cout << " Classical\n";
  
  if (check)
  { 
    memset(C2, 0, N*N*sizeof(*C2));
    MM_base<TYPE> ( A, B, C2, N); 
  }

  do 
  {
    memset(C, 0, N*N*sizeof(*C));
    if (DQSZ) // Divide & Conquer Version
      MM_DQ<TYPE> ( A, B, C, N, N); 
    else      // Classical version
      MM<TYPE> ( A, B, C, N);

    if ( check && compare<TYPE>(C,C2,N) != 0 )
    {
      std::cout << "CHECK= FAILS when remaining " << check << " repetitions" << "\n";
      break;
    }
    if (check) 
    {
      if (check == 1)
        std::cout << "CHECK= SUCCESS" << "\n";
      check--;
    }
  } while (check);

  std::cout << "Result= " << add_elements(C,N) << "\n";

  free (A);  free (B);  free (C);  free (C2);
  return 0;
}
