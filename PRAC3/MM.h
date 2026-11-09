// Classical Matrix Multiply
template <class type>
  void MM ( const type *a, const type *b, type *c, const int N )
{
  int i,j,k;
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      for (k=0; k<N; k++)
        c[i*N+j] += a[i*N+k]*b[k*N+j];
}

// Recursive, Divide & Conquer Matrix Multiply

static int DQSZ; // Smaller Size for a subproblem 

// c[][] = c[][] + a[][] * b[][]
template <class type>
  void MM_DQ ( const type *a, const type *b, type *c, int SZ, const int N)
{
  int i,j,k;
  // SZ: dimension of submatrices a, b and c. 
  // N:  size of original input matrices (size of a row)

  if (SZ <= DQSZ) 
  { // Classical algorithm for base case
    for (i=0; i<SZ; i++)
      for (j=0; j<SZ; j++)
        for (k=0; k<SZ; k++)
          c[i*N+j] += a[i*N+k]*b[k*N+j];
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
