#pragma once
#include <string.h>
#include <math.h>

/// @brief internal matrix negation function, sets contents of mem to -mem
/// @tparam T datatype of matrix
/// @param mem pointer to matrix 
/// @param len size of the matrix (rows * cols)
template <class T = float>
void __mat_neg(T *mem, int len) {
    for (int i = 0; i < len; i++) { mem[i] = -mem[i]; }
}

/// @brief internal matrix scalar multiplication
/// @tparam T datatype of matrix
/// @param mem pointer to matrix
/// @param scalar scalar to multiply matrix by
/// @param size size of the matrix
template <class T = float>
void __mat_mul_scalar(T *mem, T scalar, int size) {
    for ( int i = 0; i < size; i++ ) { mem[i] *= scalar; }
}

/// @brief internal matrix transpose function
/// @tparam T datatype of matrix
/// @param mem pointer to matrix to be transposed
/// @param ret pointer to return matrix
/// @param rows number of rows in both matrices
/// @param cols number of columns in both matrices
template <class T = float>
void __mat_transpose(const T *mem, T *ret, int rows, int cols) {
    for ( int i = 0; i < rows; i++ ) {
        int i_scalar = i * cols;
        for ( int j = 0; j < cols; j++ ) {
            ret[j * i_scalar + i] = mem[i_scalar + j];
        }
    }
}

/// @brief internal matrix row swap function
/// @tparam T datatype of matrix
/// @param mem pointer to matrix 
/// @param rows number of rows in matrix
/// @param cols number of columns in matrix
/// @param row_src source row
/// @param row_dest destination row
template <class T = float>
void __mat_swap_row(T *mem, int rows, int cols, int row_src, int row_dest) {
    T tmp;
    int col_dest = row_dest * cols;
    int col_src = row_src * cols;
    for ( int i = 0; i < cols; i++ ) {
        tmp = mem[col_dest + i];
        mem[col_dest + i] = mem[col_src + i];
        mem[col_src + i] = tmp;
    }
}

/// @brief internal matrix equality check, returns true if matrices are equal
/// @tparam T datatype of matrices
/// @param mem_lhs pointer to first matrix
/// @param mem_rhs pointer to second matrix
/// @param len size of the matrix (rows * cols)
/// @return true if the matrices are identical
template <class T = float>
bool __mat_eq(const T *mem_lhs, const T *mem_rhs, int len) {
    bool ret = true;
    for (int i = 0; i < len; i++) { ret &= (mem_lhs[i] == mem_rhs[i]); }
    return ret;
}

/// @brief internal matrix assignment operation, sets values of LHS to values of RHS
/// @tparam T datatype of matrices
/// @param mem_lhs pointer to LHS matrix
/// @param mem_rhs pointer to RHS matrix
/// @param len size of the matrix (rows * cols)
template <class T = float>
void __mat_assign(T *mem_lhs, const T *mem_rhs, int len) {
    for (int i = 0; i < len; i++) { mem_lhs[i] = mem_rhs[i]; }
}

/// @brief internal matrix subtraction function, subtracts RHS from LHS
/// @tparam T datatype of matrices
/// @param mem_lhs pointer to LHS of operation 
/// @param mem_rhs pointer to RHS of operation
/// @param len size of the matrix (rows * cols)
template <class T = float>
void __mat_sub(T *mem_lhs, const T *mem_rhs, int len) {
    for (int i = 0; i < len; i++) { mem_lhs[i] -= mem_rhs[i]; }
}

/// @brief internal matrix subtraction function, adds RHS to LHS
/// @tparam T datatype of matrices
/// @param mem_lhs pointer to LHS of operation 
/// @param mem_rhs pointer to RHS of operation
/// @param len size of the matrix (rows * cols)
template <class T = float>
void __mat_add(T *mem_lhs, const T *mem_rhs, int len) {
    for (int i = 0; i < len; i++) { mem_lhs[i] += mem_rhs[i]; }
}

/// @brief internal matrix multiplication function, multiplies LHS by RHS, 
/// @tparam T datatype of matrices
/// @param mem_lhs pointer to LHS of operation 
/// @param mem_rhs pointer to RHS of operation
/// @param mem_ret pointer to the output matrix
/// @param rows_lhs numer of rows in mem_lhs
/// @param cols_lhs numer of cols in mem_lhs
/// @param rows_rhs numer of rows in mem_rhs
/// @param cols_rhs numer of cols in mem_rhs
template <class T = float>
void __mat_mul(const T* mem_lhs, const T* mem_rhs, T* mem_ret, int rows_lhs, int cols_lhs, int rows_rhs, int cols_rhs) {
    
    for ( int i = 0; i < rows_rhs; i++ ) {
        int i_scalar_lhs = i*cols_lhs;
        int i_scalar_rhs = i*cols_rhs;
        for ( int j = 0; j < cols_rhs; j++ ) {
            T sum = mem_lhs[i_scalar_lhs] * mem_rhs[j];
            for ( int k = 1; k < cols_lhs; k++ ) {
                // skip multiplication if multiplying by zero
                #ifdef attempt_faster_mul
                    if ( ((*(int*)&mem_lhs[i_scalar_lhs+k]) & 0x7fffffff) != 0 ) {
                        sum += mem_lhs[i_scalar_lhs + k] * mem_rhs[k * cols_rhs + j];
                    }
                #else
                    sum += mem_lhs[i_scalar_lhs + k] * mem_rhs[k * cols_rhs + j];
                #endif
            }
            mem_ret[i_scalar_rhs + j] = sum;
        }
    }
}

/// @brief internal matrix inversion function using LU decomposition. https://en.wikipedia.org/wiki/LU_decomposition
/// @tparam T datatype of matrices
/// @param mem pointer to matrix to be inverted
/// @param A pointer to temporary matrix used in the calculation
/// @param ret pointer to return matrix
/// @param rows number of rows in all matrices
/// @param cols number of columns in all matrices 
/// @param tolerance tolerance of the equation
/// @return true if the LU decomposition was a success
template <class T = float>
bool __mat_inv(T *mem, T *A, T *ret, int rows, int cols, T tolerance = 1e-9) {

    int mat_size = rows * cols;

    __mat_assign(A, mem, mat_size);

    int P[rows + 1];
    for ( int i = 0; i < rows; i++ ) { P[i] = i; }

    int i, j, k, imax;
    T max_a, abs_a;

    for ( i = 0; i < rows; i++ ) {

        max_a = 0;
        imax = i;

        for ( k = i; k < rows; k++ )  {
            abs_a = (A[k * cols + i]);
            abs_a = abs_a < 0 ? -abs_a : abs_a;
            if ( abs_a > max_a ) {
                max_a = abs_a;
                imax = k;
            }
        }

        if ( max_a < tolerance ) {
            return false;
        }

        if ( imax != i ) {
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            __mat_swap_row(A, rows, cols, i, imax);

            P[rows]++;
        }

        for ( j = i + 1; j < rows; j++ ) {
            int j_cols = j * cols;
            int i_cols = i * cols;
            A[j_cols + i] /= A[i_cols + i];
            for ( k = i+1; k < rows; k++ ) {
                A[j_cols + k] -= A[j_cols + i] * A[i_cols + k];
            }
        }
    }

    for ( j = 0; j < rows; j++ ) {

        for ( i = 0; i < rows; i++ ) {

            int i_cols = i * cols;
            ret[i_cols + j] = (P[i] == j);

            for ( k = 0; k < i; k++ ) { ret[i_cols + j] -= A[i_cols + k] * ret[k * cols + j]; }

        }

        for ( i = rows - 1; i >= 0; i-- ) {

            int i_cols = i * cols;
            
            for ( k = i + 1; k < rows; k++ ) { ret[i_cols + j] -= A[i_cols + k] * ret[k * cols + j]; }

            ret[i_cols + j] /= A[i_cols + i];

        }

    }

    return true;
}

template <class T = float, int _rows = 1, int _cols = 1>
class matrix {

public:

    const static int rows = _rows;
    const static int cols = _cols;
    const static int mem_size = rows * cols;

    T mem[rows * cols];

    // no clue how this works, but it does
    matrix() = default;

    
	// element access and assignment
	T  operator()(int row, int col = 0) const { return(mem[(row * cols + col)]); }
	T &operator()(int row, int col = 0)       { return(mem[(row * cols + col)]); }

    bool swap_row(int row_src, int row_dest) {
        if ( row_src < 0 || row_dest < 0 || row_src > rows || row_dest > rows ) { return false; }
        __mat_swap_row(mem, rows, cols, row_src, row_dest);
        return true;
    }

	matrix &operator=(const matrix &other) {
		static_assert(rows == other.rows && cols == other.cols, "Matrices are not of equal size!");
		memcpy(mem, other.mem, (rows*cols)*sizeof(T));
		return *this;
	}

	matrix operator-() {
		__mat_neg(mem, mem_size);
		return *this;
	}

	matrix &operator+=(const matrix &other) {
		static_assert(rows == other.rows && cols == other.cols, "Matrices are not of equal size!");
		__mat_add(mem, other.mem, mem_size);
		return *this;
	}

	const matrix operator+(const matrix &other) const {
		return matrix(*this) += other;
	}

	matrix &operator-=(const matrix &other) {
		static_assert(rows == other.rows && cols == other.cols, "Matrices are not of equal size!");
		__mat_sub(mem, other.mem, mem_size);
		return *this;
	}

	const matrix operator-(const matrix &other) const {
		return matrix(*this) -= other;
	}
	
	template<int oprows = 1, int opcols = 1>
	const matrix<T, rows, oprows> operator*(const matrix<T, opcols, oprows> &other) const {
        static_assert(cols == other.rows, "Matrices are not of equal size!");
		matrix<T, opcols, oprows> ret;
        __mat_mul(mem, other.mem, ret.mem, rows, cols, other.rows, other.cols);
		return ret;
	}

	matrix &operator*=(T scalar) {
		__mat_mul_scalar(mem, scalar, mem_size);
		return *this;
	}

	matrix operator~() {
		matrix<T, rows, cols> ret;
		__mat_transpose(mem, ret.mem, rows, cols);
		return ret;
	}

    matrix invert(bool *ok, T tol = 1e-9) {
        static_assert(rows == cols, "Cannot invert a non-square matrix");
        matrix<T, rows, cols> A;
        matrix<T, rows, cols> ret;
        *ok = __mat_inv(mem, A.mem, ret.mem, rows, cols, tol);
        return ret;
    }

    matrix mask(bool mask[rows*cols]) {
        matrix<T, rows, cols> ret;
        for ( int i = 0; i < mem_size; i++ ) {
            if ( mask[i] ) { ret.mem[i] = mem[i]; } else { ret.mem[i] = 0; }
        }
        return ret;
    }

    matrix mask(const matrix<bool, rows, cols> &mask) {
        matrix<T, rows, cols> ret;
        for ( int i = 0; i < mem_size; i++ ) {
            if ( mask.mem[i] ) { ret.mem[i] = mem[i]; } else { ret.mem[i] = 0; }
        }
        return ret;
    }

};