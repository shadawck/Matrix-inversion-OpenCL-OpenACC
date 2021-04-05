#include "Matrix.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include "Chrono.hpp"
#include "../../../../../usr/include/c++/10/valarray"

using namespace std;

void findPivot(int row, MatrixConcatCols &augmentedMatrix, int &pivot);

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix);

double *convertValArrayToDouble(valarray<double> array);

void checkSingularity(const MatrixConcatCols &augmentedMatrix, int p, int k);

Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2);

void printResult(int matrixDimension, Chrono cron, Matrix &lRes);

void printAugMatrix1DArray(double *matrix, int size);


void arrayToMatrix(MatrixConcatCols &augmentedMatrix, double *augMat, Matrix &resMatrix);

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
 * OPENACC implementation
 * @param mat Original matrix
 */
void invertParallel(Matrix &mat) {
    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));
    int rowSize = augmentedMatrix.rows();
    int colSize = augmentedMatrix.cols();

    for (int row = 0; row < mat.rows(); ++row) {
        int pivot = row;
        double lMax = fabs(augmentedMatrix(row, row));

//        Chrono cron1 = Chrono(true);
        for (int i = row; i < rowSize; ++i) {
            if (fabs(augmentedMatrix(i, row)) > lMax) {
                lMax = fabs(augmentedMatrix(i, row));
                pivot = i;
            }
        }

//        cron1.pause();
//        cout << "Cron1 : " << cron1.get() << endl;

        if (augmentedMatrix(pivot, row) == 0) {
            throw runtime_error("Matrix is not invertible");
        }

        if (pivot != row) {
            augmentedMatrix.swapRows(pivot, row);
        }

        double pivotValue = augmentedMatrix(row, row);

//        Chrono cron2 = Chrono(true);

        for (int col = 0; col < colSize; ++col) {
            augmentedMatrix(row, col) /= pivotValue;
        }
//        cron2.pause();
//        cout << "Cron2 : " << cron2.get() << endl;

//      Chrono cron3 = Chrono(true);


        for (int i = 0; i < rowSize; ++i) {
            if (i != row) {
                double llValue = augmentedMatrix(i, row);
                augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(row) * llValue;
            }
        }

//        cron3.pause();
//        cout << "Cron3 : " <<  cron3.get() << endl;
    }


//    Chrono cron4 = Chrono(true);
    for (int i = 0; i < mat.rows(); ++i) {
        mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * colSize + mat.cols(), mat.cols(), 1)];
    }
//    cron4.pause();
//    cout << "Cron4 : " << cron4.get() << endl;
}

void invertParallelRaw(double *augMat, int size) {
    int colSize = size * 2; // number of column and size of a row

    for (int row = 0; row < size; ++row) {
        int pivot = row;
        double lMax = fabs(augMat[colSize * row + row]);

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
            double tmpRow[colSize];
            for (int i = 0; i < colSize; ++i) {
                tmpRow[i] = augMat[colSize * row + i];
                augMat[colSize * row + i] = augMat[colSize * pivot + i];
                augMat[colSize * pivot + i] = tmpRow[i];
            }
        }

        double pivotVal = augMat[colSize * row + row];
        for (int i = 0; i < colSize; ++i) {
            augMat[colSize * row + i] /= pivotVal;
        }

        double rowCopy[colSize];
        for (int i = 0; i < size; ++i) {
            if (i != row) {
                double llValue = augMat[colSize * i + row];
                // get row of index "row"
                for (int j = 0; j < colSize; ++j) {
                    rowCopy[j] = augMat[colSize * row + j] * llValue;
                    augMat[colSize * i + j] -= rowCopy[j];  // substitution on i slice
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
    /**
    * Sequential execution
    */
    cout << "--- SEQUENTIAL EXECUTION ---" << endl;
    Matrix seqMatrix(randomMatrix);

    auto cronSeq = Chrono(true);
    invertSequential(seqMatrix);
    cronSeq.pause();

    Matrix lResSeq = multiplyMatrix(seqMatrix, copyRandomMatrix);
    printResult(matrixDimension, cronSeq, lResSeq);

    /**
     * openACC execution
     */
    cout << endl << " --- PARALLEL EXECUTION --- " << endl;

    Matrix parMatrix = Matrix(randomMatrix);
    MatrixConcatCols augmentedMatrix(parMatrix, MatrixIdentity(parMatrix.rows()));

    double *augMat = convertValArrayToDouble(augmentedMatrix.getDataArray());

    auto cronPar = Chrono(true);
    invertParallelRaw(augMat, augmentedMatrix.rows());
    cronPar.pause();

    Matrix resMatrix(matrixDimension, matrixDimension);
    arrayToMatrix(augmentedMatrix, augMat, resMatrix);

    Matrix lResPar = multiplyMatrix(resMatrix, copyRandomMatrix);
    printResult(matrixDimension, cronPar, lResPar);

    return 0;
}

void arrayToMatrix(MatrixConcatCols &augmentedMatrix,  double *augMat,  Matrix &resMatrix) {
    for(int i = 0; i < augmentedMatrix.cols() * augmentedMatrix.cols(); i++){
        augmentedMatrix.getDataArray()[i] = augMat[i];
    }

    for (int i = 0; i < resMatrix.rows(); ++i) {
        resMatrix.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * resMatrix.cols()*2 + resMatrix.cols(), resMatrix.cols(), 1)];
    }
}

void printResult(int matrixDimension, Chrono cron, Matrix &lRes) {
    cout << "Matrix dimension : " << matrixDimension << endl;
    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
    cout << "Total execution time : " << cron.get() << endl;
}

double *convertValArrayToDouble(valarray<double> array) {
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
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