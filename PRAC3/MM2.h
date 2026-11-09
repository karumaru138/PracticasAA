// Classical Matrix Multiply
template <class type>
  void MM ( const type *a, const type *b, type *c, const int N )
{
  for (int i=0; i<N; i+=2)
    for (int k=0; k<N; k+=2)
      for (int j=0; j<N; j++)
      {
          type B1 = b[k*N+j];
          type B2 = b[(k+1)*N+j];

          c[  i  *N+j] += a[  i  *N+k]*B1 + a[  i  *N+k+1]*B2;
          c[(i+1)*N+j] += a[(i+1)*N+k]*B1 + a[(i+1)*N+k+1]*B2;
      }
}

// Recursive, Divide & Conquer Matrix Multiply

static int DQSZ; // Smaller Size for a subproblem 

// c[][] = c[][] + a[][] * b[][]
template <class type>
void MM_DQ ( const type *a, const type *b, type *c, int SZ, const int N)
{
  // SZ: dimension of submatrices a, b and c. 
  // N:  size of original input matrices (size of a row)

  if (SZ <= DQSZ) 
  { // Classical algorithm for base case
    for (int i=0; i<SZ; i+=2)
      for (int k=0; k<SZ; k+=2)
        for (int j=0; j<SZ; j++)
        {
          type B1 = b[k*N+j];
          type B2 = b[(k+1)*N+j];

          c[  i  *N+j] += a[  i  *N+k]*B1 + a[  i  *N+k+1]*B2;
          c[(i+1)*N+j] += a[(i+1)*N+k]*B1 + a[(i+1)*N+k+1]*B2;
        }
    return;
  }

  // Divide task into 8 subtasks

  SZ = SZ/2;  // assume SZ is a perfect power of 2

  MM_DQ<type> ( a,          b,          c,          SZ, N);
  MM_DQ<type> ( a,          b+SZ,       c+SZ,       SZ, N);
  MM_DQ<type> ( a+SZ*N,     b,          c+SZ*N,     SZ, N);
  MM_DQ<type> ( a+SZ*N,     b+SZ,       c+SZ*(N+1), SZ, N);

  MM_DQ<type> ( a+SZ,       b+SZ*N,     c,          SZ, N);
  MM_DQ<type> ( a+SZ,       b+SZ*(N+1), c+SZ,       SZ, N);
  MM_DQ<type> ( a+SZ*(N+1), b+SZ*N,     c+SZ*N,     SZ, N);
  MM_DQ<type> ( a+SZ*(N+1), b+SZ*(N+1), c+SZ*(N+1), SZ, N);
}
