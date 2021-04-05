#define blocksize 8



__kernel void vecadd(__global int *A,
                  __global int *C) {

   int idx = get_global_id(0);
   int idy = get_global_id(1);

   printf("%d : %d\n", idx, idy);
   printf("%d", get_work_dim());

   C[idx] = A[idx];

}
