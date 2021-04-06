
__kernel void inversion(__global double *mat,
                        __global double *eyeResMat,
                        int size,
                        int index) {

    int idx = get_global_id(0);

//    for (int index = 0; index < size; ++index) {

    double scale = 1.0 / mat[size * index + index]; // diag

    mat[size * index + idx] *= scale;
    eyeResMat[size * index + idx] *= scale;


    if (idx != index) {
        double currentScale = mat[size * idx + index];

        for (int j = 0; j < size; ++j) {
            mat[size * idx + j] = mat[size * idx + j] - currentScale * mat[size * index + j];
            eyeResMat[size * idx + j] = eyeResMat[size * idx + j] - currentScale * eyeResMat[size * index + j];
        }
    }
}


