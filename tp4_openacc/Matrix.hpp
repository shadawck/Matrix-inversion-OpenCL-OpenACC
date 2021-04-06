//
//  Matrix.cpp
//

#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__

#include <string>
#include <valarray>
#include <cassert>
#include <iostream>

using namespace std;

// La classe Matrix est dérivée de std::valarray. Cette dernière est
// similaire à std::vector, sauf qu'elle alloue exactement la quantité
// de mémoire nécessaire (au lieu de 2^n dans std::vector) et qu'elle
// contient des fonctions pour accéder facilement à des régions (splice)
// du vecteur.
class Matrix {

public:

    // Construire matrice iRows x iCols et initialiser avec des 0.
    Matrix(int iRows, int iCols) : mRows(iRows), mCols(iCols), mData(0., iRows * iCols) {}

    // Affecter une matrice de même taille; s'assurer que les tailles sont identiques.

    // Accéder à la case (i, j) en lecture/écriture.
    inline double &operator()(int iRow, int iCol) {
        return mData[(iRow * mCols) + iCol];
    }

    // Accéder à la case (i, j) en lecture seulement.
    inline const double &operator()(int iRow, int iCol) const {
        return mData[(iRow * mCols) + iCol];
    }

    // Retourner le nombre de colonnes.
    inline int cols() const { return mCols; }

    // Retourner le nombre de lignes.
     inline int rows() const { return mRows; }

    // Retourner le tableau d'une colonne de la matrice.
     valarray<double> getColumnCopy(int iCol) const {
        assert(iCol < mCols);
        return mData[slice(iCol, mRows, mCols)];
    }

    // Retourner la slice d'une colonne de la matrice.
    slice_array<double> getColumnSlice(int iCol) {
        assert(iCol < mCols);
        return mData[slice(iCol, mRows, mCols)];
    }

    // Retourner la slice d'une colonne de la matrice.
     slice_array<double> getColumnSlice(int iCol) const {
        assert(iCol < mCols);
        return const_cast<Matrix *>(this)->mData[slice(iCol, mRows, mCols)];
    }

    // Retourner le tableau d'une rangée de la matrice.
     valarray<double> getRowCopy(int iRow) const {
        assert(iRow < mRows);
        return mData[slice(iRow * mCols, mCols, 1)];
    }

    // Retourner la slice d'une rangée de la matrice.
    slice_array<double> getRowSlice(int iRow) {
        assert(iRow < mRows);
        return mData[slice(iRow * mCols, mCols, 1)];
    }

    // Retourner la slice d'une rangée de la matrice.
     slice_array<double> getRowSlice(int iRow) const {
        assert(iRow < mRows);
        return const_cast<Matrix *>(this)->mData[slice(iRow * mCols, mCols, 1)];
    }

    // Accéder au tableau interne de la matrice en lecture/écriture.
    valarray<double> &getDataArray() { return mData; }

    // Accéder au tableau interne de la matrice en lecture seulement.
     const valarray<double> &getDataArray() const { return mData; }

    // Permuter deux rangées de la matrice.
    Matrix &swapRows(int iR1, int iR2);

    // Permuter deux colonnes de la matrice.
    Matrix &swapColumns(int iC1, int iC2);

    // Représenter la matrice sous la forme d'une chaîne de caractères.
    // Pratique pour le débuggage...
     string str() const;

protected:
    // Nombre de rangées et de colonnes.
    int mRows, mCols;
    valarray<double> mData;
};

// Construire une matrice identité.
class MatrixIdentity : public Matrix {
public:
    explicit MatrixIdentity(int iSize);
};

// Construire une matrice aléatoire [0,1) iRows x iCols.
// Utiliser srand pour initialiser le générateur de nombres.
class MatrixRandom : public Matrix {
public:
    MatrixRandom(int iRows, int iCols);
};

class MatrixExample : public Matrix {
public:
    MatrixExample(int iRows, int iCols);
};

// Construire une matrice en concaténant les colonnes de deux matrices de même hauteur.
class MatrixConcatCols : public Matrix {
public:
    MatrixConcatCols(const Matrix &iMat1, const Matrix &iMat2);
};

// Construire une matrice en concaténant les rangées de deux matrices de même largeur.
class MatrixConcatRows : public Matrix {
public:
    MatrixConcatRows(const Matrix &iMat1, const Matrix &iMat2);
};

// Insérer une matrice dans un flot de sortie.
ostream &operator<<(ostream &oStream, const Matrix &iMat);

#endif
