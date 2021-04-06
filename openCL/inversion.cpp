__kernel void inversion(__global double *mat,
                        __global double *eyeResMat,
                        int size) {

    int idx = get_global_id(0);

    if (idx == 0) {
        printf("ARGUMENT IN DEVICE\n");
        printf("Size : %d\n", size);
        printf("First mat element : %f  \n", mat[0]);
        printf("First element of Eye Matrix : %f\n", eyeResMat[0]);
    }

    for (int row = 0; row < size; ++row) {
        double scale = 1.0 / mat[size * row + row]; // diag

        printf("scale : %f\n", scale);

        for (int i = 0; i < size; ++i) {
            mat[size * row + i] *= scale;
            eyeResMat[size * row + i] *= scale;
        }

        for (int i = 0; i < size; ++i) {
            if (i != row) {
                double currentScale = mat[size * i + row];

                for (int j = 0; j < size; ++j) {
                    mat[size * i + j] = mat[size * i + j] - currentScale * mat[size * row + j];
                    eyeResMat[size * i + j] = eyeResMat[size * i + j] - currentScale * eyeResMat[size * row + j];
                }
            }
        }
    }
}

