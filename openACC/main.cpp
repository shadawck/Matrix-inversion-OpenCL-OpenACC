#include "Matrix.hpp"
#include "Chrono.hpp"

using namespace std;

double *convertValArrayToDouble(valarray<double> array);

double **MatrixTo2DArray(Matrix mat);

Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2);

void printResult(int matrixDimension, Chrono cron, Matrix &lRes);

void printResultMin(int matrixDimension, Chrono cron);

void arrayToMatrix(MatrixConcatCols &augmentedMatrix, const double *augMat, const Matrix &resMatrix);

Matrix multiArrayToMatrix(double **multiArray, int height, int width);

void cleanArray(double **multiArray, int size);

/// DEBUG ///
void printAugMatrix1DArray(double *matrix, int size);

/**
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
 * @param mat
 */
void invertSequential(Matrix &mat) {
    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));
    int rowSize = augmentedMatrix.rows();
    int colSize = augmentedMatrix.cols();

    {
        for (int row = 0; row < mat.rows(); ++row) {

            int pivot = row;
            double lMax = fabs(augmentedMatrix(row, row));

            for (int i = row; i < rowSize; ++i) {
                if (fabs(augmentedMatrix(i, row)) > lMax) {
                    lMax = fabs(augmentedMatrix(i, row));
                    pivot = i;
                }
            }

            if (augmentedMatrix(pivot, row) == 0) {
                throw runtime_error("Matrix is not invertible");
            }

            if (pivot != row) {
                augmentedMatrix.swapRows(pivot, row);
            }

            double pivotValue = augmentedMatrix(row, row);

            for (int i = 0; i < colSize; ++i) {
                augmentedMatrix(row, i) /= pivotValue;
            }
            for (int i = 0; i < rowSize; ++i) {
                if (i != row) {
                    double llValue = augmentedMatrix(i, row);
                    augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(row) * llValue;
                }
            }
        }
    }

    for (int i = 0; i < mat.rows(); ++i) {
        mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * colSize + mat.cols(), mat.cols(), 1)];
    }
}


/**
 * Invert matrix with Gauss Jordan method
 * Raw (without Matrix class) && OPENACC implementation
 * @param mat Original matrix
 */
void invertParallelRaw(double *augMat, int size) {
    int colSize = size * 2; // number of column and size of a row

    // Use static array so dont need to deallocate. Will be release after loop

    for (int row = 0; row < size; ++row) {
#pragma acc kernels
        {
            int pivot = row;
            double lMax = fabs(augMat[colSize * row + row]);

#pragma acc for independent
            for (int i = row; i < size; ++i) {
                if (fabs(augMat[colSize * i + row]) > lMax) {
                    lMax = fabs(augMat[colSize * i + row]);
                    pivot = i; // row of pivot
                }
            }

            if (augMat[colSize * row + row] == 0) {
                throw runtime_error("Matrix is not invertible");
            }

            if (pivot != row) {
#pragma acc for present(augMat[0:size*colSize])
                for (int i = 0; i < colSize; ++i) {
                    /// Arithetic operation
                    augMat[colSize * row + i] = augMat[colSize * row + i] + augMat[colSize * pivot + i];
                    augMat[colSize * pivot + i] = augMat[colSize * row + i] - augMat[colSize * pivot + i];
                    augMat[colSize * row + i] = augMat[colSize * row + i] - augMat[colSize * pivot + i];
                }
            }

            double pivotVal = augMat[colSize * row + row];

#pragma acc for independent
            for (int i = 0; i < colSize; ++i) {
                augMat[colSize * row + i] /= pivotVal;
            }

#pragma acc for independent
            for (int i = 0; i < size; ++i) {
                if (i != row) {
                    double llValue = augMat[colSize * i + row];
                    for (int j = 0; j < colSize; ++j) { // get row of index "row"
                        augMat[colSize * i + j] -= augMat[colSize * row + j] * llValue;// substitution on i slice
                    }
                }
            }
        }
    }
}

void bruteForce(double **mat, double **eyeMat, int size) {

    for (int fd = 0; fd < size; ++fd) {
#pragma acc kernels
        {
            double scale = 1.0 / mat[fd][fd]; // diag

#pragma acc for independent
            for (int i = 0; i < size; ++i) {
                mat[fd][i] *= scale;
                eyeMat[fd][i] *= scale;
            }

#pragma acc for independent
            for (int i = 0; i < size; ++i) {
                if (i != fd) {
                    double crScale = mat[i][fd];
#pragma acc for independent
                    for (int j = 0; j < size; ++j) {
                        mat[i][j] = mat[i][j] - crScale * mat[fd][j];
                        eyeMat[i][j] = eyeMat[i][j] - crScale * eyeMat[fd][j];
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {

    srand((unsigned) time(nullptr));

    int matrixDimension = 5;
    if (argc == 2) {
        matrixDimension = atoi(argv[1]);
    }

    MatrixRandom randomMatrix(matrixDimension, matrixDimension);
    const Matrix &copyRandomMatrix(randomMatrix);

//    /**
//    * Sequential execution
//    */
//    cout << "--- SEQUENTIAL EXECUTION ---" << endl;
//    Matrix seqMatrix(randomMatrix);
//
//    auto cronSeq = Chrono(true);
//    invertSequential(seqMatrix);
//    cronSeq.pause();
//
////    Matrix lResSeq = multiplyMatrix(seqMatrix, copyRandomMatrix);
////    printResult(matrixDimension, cronSeq, lResSeq);
//    printResultMin(matrixDimension,cronSeq);
//
//    /**
//     * openACC NUMERO 1 execution
//     */
//    cout << endl << " --- PARALLEL EXECUTION SOLUTION 1 --- " << endl;
//
    Matrix parMatrix = Matrix(randomMatrix);
//    MatrixConcatCols augmentedMatrix(parMatrix, MatrixIdentity(parMatrix.rows()));
//
//    auto *augMat = (double *) malloc(matrixDimension * matrixDimension * 2 * sizeof(double));
//    augMat = convertValArrayToDouble(augmentedMatrix.getDataArray());
//
//    auto cronPar = Chrono(true);
//    invertParallelRaw(augMat, augmentedMatrix.rows());
//    cronPar.pause();
//
////    Matrix resMatrix(matrixDimension, matrixDimension);
////    arrayToMatrix(augmentedMatrix, augMat, resMatrix);
////    cout << " -- Calculating Error Solution 1 --" << endl;
////    Matrix lResPar = multiplyMatrix(resMatrix, copyRandomMatrix);
////    printResult(matrixDimension, cronPar, lResPar);
//    printResultMin(matrixDimension,cronPar);


    /**
     * openACC SOLUTION 2 execution
     */

    cout << endl << " --- PARALLEL EXECUTION SOLUTION 2 --- " << endl;

    double **newMat = MatrixTo2DArray(parMatrix);
    double **eyeMat = MatrixTo2DArray(MatrixIdentity(parMatrix.rows()));

    auto cronPar_2 = Chrono(true);
    bruteForce(newMat, eyeMat, parMatrix.rows());
    cronPar_2.pause();

    cout << " -- Calculating Error Solution 2 --" << endl;
    Matrix resMatrix_2 = multiArrayToMatrix(eyeMat, parMatrix.rows(), parMatrix.rows());
    Matrix lResPar_2 = multiplyMatrix(resMatrix_2, copyRandomMatrix);
    printResult(matrixDimension, cronPar_2, lResPar_2);
    printResultMin(matrixDimension,cronPar_2);

    cout << " -- De-allocation (cleaning) --" << endl;

//    delete[] augMat;
    cleanArray(eyeMat, parMatrix.rows());
    cleanArray(newMat, parMatrix.rows());

    return 0;
}

void arrayToMatrix(MatrixConcatCols &augmentedMatrix, const double *augMat, const Matrix &resMatrix) {
    for (int i = 0; i < augmentedMatrix.cols() * augmentedMatrix.rows(); i++) {
        augmentedMatrix.getDataArray()[i] = augMat[i];
    }

    for (int i = 0; i < resMatrix.rows(); ++i) {
        resMatrix.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * resMatrix.cols() * 2 + resMatrix.cols(),
                                                                        resMatrix.cols(), 1)];
    }
}

void printResult(int matrixDimension, Chrono cron, Matrix &lRes) {
    cout << "Matrix dimension : " << matrixDimension << endl;
    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
    cout << "Total execution time : " << cron.get() << endl;
}

void printResultMin(int matrixDimension, Chrono cron) {
    cout << "Matrix dimension : " << matrixDimension << endl;
    cout << "Total execution time : " << cron.get() << endl;
}

double *convertValArrayToDouble(valarray<double> array) {
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
}

double **MatrixTo2DArray(Matrix mat) {
    /// Allocate
    auto **newArray = (double **) malloc(mat.rows() * sizeof(double *));
    for (int i = 0; i < mat.rows(); i++)
        newArray[i] = (double *) malloc(mat.cols() * sizeof(double));

    for (int i = 0; i < mat.rows(); i++)
        for (int j = 0; j < mat.cols(); j++)
            newArray[i][j] = mat(i, j);
    return newArray;
}

Matrix multiArrayToMatrix(double **multiArray, int height, int width) {
    Matrix newMatrix(height, width);
    for (int i = 0; i < height; i++)
        for (int j = 0; j < height; j++)
            newMatrix(i, j) = multiArray[i][j];

    return newMatrix;
}

void cleanArray(double **multiArray, int size) {
    for (int i = 0; i < size; i++)
        delete[] multiArray[i];
    delete[] multiArray;
}


Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2) {
    assert(iMat1.cols() == iMat2.rows());
    Matrix lRes(iMat1.rows(), iMat2.cols());
    for (int i = 0; i < lRes.rows(); ++i) {
        for (int j = 0; j < lRes.cols(); ++j) {
            lRes(i, j) = (iMat1.getRowCopy(i) * iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

/// DEBUG ///
void printAugMatrix1DArray(double *matrix, int size) {
    int colSize = size * 2;
    cout << "[" << endl;
    for (int i = 0; i < size; ++i) {
        cout << "   [ ";
        for (int j = 0; j < colSize; ++j) {
            cout << matrix[colSize * i + j] << ", ";
        }
        cout << "]," << endl;
    }
    cout << "]" << endl;
}
