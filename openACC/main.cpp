#include "Matrix.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include "Chrono.hpp"

using namespace std;

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot);

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix);

double *convertValArrayToDouble(valarray<double> array);

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t p, size_t k);

Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2);

void printResult(int matrixDimension, Chrono cron, Matrix &lRes);


/**
 * Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
 * @param mat
 */
void invertSequential(Matrix &mat) {
    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));

    // traiter chaque rangée
    for (size_t row = 0; row < mat.rows(); ++row) {

        size_t pivot;
        findPivot(row, augmentedMatrix, pivot);
        checkSingularity(augmentedMatrix, pivot, row);

        // échanger la index courante avec celle du pivot
        if (pivot != row) augmentedMatrix.swapRows(pivot, row);

        double pivotValue = augmentedMatrix(row, row);

        for (size_t col = 0; col < augmentedMatrix.cols(); ++col) {
            // On divise les éléments de la rangée index par la valeur du pivot.
            // Ainsi, augmentedMatrix(index,index) deviendra égal à 1.
            augmentedMatrix(row, col) /= pivotValue;
        }

        for (size_t i = 0; i < augmentedMatrix.rows(); ++i) {         // Pour chaque rangée...
            if (i != row) { // ...différente de index
                double llValue = augmentedMatrix(i, row);
                // On soustrait la rangée index multipliée par l'élément index de la rangée courante
                augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(row) * llValue;
            }
        }
    }
    // On copie la partie droite de la matrice AI ainsi transformée dans la matrice courante (this).
    splitAugmentedMatrix(mat, augmentedMatrix);
}

/**
 * Invert matrix with Gauss Jordan method
 * @param mat Original matrix
 */
void invertParallel(Matrix &mat) {
    /* OPENACC implementation */
    assert(mat.rows() == mat.cols());
    MatrixConcatCols augmentedMatrix(mat, MatrixIdentity(mat.rows()));

    // traiter chaque rangée
    for (size_t row = 0; row < mat.rows(); ++row) {

        size_t pivot;
        findPivot(row, augmentedMatrix, pivot);
        checkSingularity(augmentedMatrix, pivot, row);

        // échanger la index courante avec celle du pivot
        if (pivot != row) augmentedMatrix.swapRows(pivot, row);

        double pivotValue = augmentedMatrix(row, row);

        for (size_t col = 0; col < augmentedMatrix.cols(); ++col) {
            // On divise les éléments de la rangée index par la valeur du pivot.
            // Ainsi, augmentedMatrix(index,index) deviendra égal à 1.
            augmentedMatrix(row, col) /= pivotValue;
        }

        for (size_t i = 0; i < augmentedMatrix.rows(); ++i) {         // Pour chaque rangée...
            if (i != row) { // ...différente de index
                double llValue = augmentedMatrix(i, row);
                // On soustrait la rangée index multipliée par l'élément index de la rangée courante
                augmentedMatrix.getRowSlice(i) -= augmentedMatrix.getRowCopy(row) * llValue;
            }
        }
    }
    // On copie la partie droite de la matrice AI ainsi transformée dans la matrice courante (this).
    splitAugmentedMatrix(mat, augmentedMatrix);
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

    auto cronPar = Chrono(true);
    invertParallel(parMatrix);
    cronPar.pause();

    Matrix lResPar = multiplyMatrix(parMatrix, copyRandomMatrix);
    printResult(matrixDimension, cronPar, lResPar);

    return 0;
}

void printResult(int matrixDimension, Chrono cron, Matrix &lRes) {
    cout << "Matrix dimension : " << matrixDimension << endl;
    cout << "Erreur: " << lRes.getDataArray().sum() - matrixDimension << endl;
    cout << "Total execution time : " << cron.get() << endl;
}

void splitAugmentedMatrix(Matrix &mat, MatrixConcatCols &augmentedMatrix) {
    for (size_t i = 0; i < mat.rows(); ++i) {
        mat.getRowSlice(i) = augmentedMatrix.getDataArray()[slice(i * augmentedMatrix.cols() + mat.cols(), mat.cols(),
                                                                  1)];
    }
}

void findPivot(size_t row, MatrixConcatCols &augmentedMatrix, size_t &pivot) {
    pivot = row;
    double lMax = fabs(augmentedMatrix(row, row));
    for (size_t i = row; i < augmentedMatrix.rows(); ++i) {
        if (fabs(augmentedMatrix(i, row)) > lMax) {
            lMax = fabs(augmentedMatrix(i, row));
            pivot = i;
        }
    }
}

double *convertValArrayToDouble(valarray<double> array) {
    auto *newArray = new double[array.size()];
    copy(begin(array), end(array), newArray);
    return newArray;
}

void checkSingularity(const MatrixConcatCols &augmentedMatrix, size_t p, size_t k) {
    if (augmentedMatrix(p, k) == 0) {
        throw runtime_error("Matrix is not invertible");
    }
}

Matrix multiplyMatrix(const Matrix &iMat1, const Matrix &iMat2) {
    assert(iMat1.cols() == iMat2.rows());
    Matrix lRes(iMat1.rows(), iMat2.cols());
    for (size_t i = 0; i < lRes.rows(); ++i) {
        for (size_t j = 0; j < lRes.cols(); ++j) {
            lRes(i, j) = (iMat1.getRowCopy(i) * iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}