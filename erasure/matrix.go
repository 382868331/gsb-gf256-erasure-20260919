package erasure

import "errors"

// errSingular is returned when a matrix over GF(256) has no inverse.
var errSingular = errors.New("erasure: matrix is singular")

// matMul returns a*b for row-major matrices over GF(256).
func matMul(a, b [][]byte) [][]byte {
	rows := len(a)
	cols := len(b[0])
	inner := len(b)
	out := make([][]byte, rows)
	for i := range out {
		out[i] = make([]byte, cols)
		for j := 0; j < cols; j++ {
			var acc byte
			for t := 0; t < inner; t++ {
				acc ^= gfMul(a[i][t], b[t][j])
			}
			out[i][j] = acc
		}
	}
	return out
}

// matInvert returns the inverse of the square matrix m over GF(256) using
// Gauss-Jordan elimination, or errSingular if no inverse exists.
// The input matrix is not modified.
func matInvert(m [][]byte) ([][]byte, error) {
	n := len(m)
	// Augment [m | I].
	aug := make([][]byte, n)
	for i := range aug {
		aug[i] = make([]byte, 2*n)
		copy(aug[i], m[i])
		aug[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		// Find a pivot row with a nonzero entry in this column.
		pivot := -1
		for r := col; r < n; r++ {
			if aug[r][col] != 0 {
				pivot = r
				break
			}
		}
		if pivot < 0 {
			return nil, errSingular
		}
		aug[col], aug[pivot] = aug[pivot], aug[col]
		// Normalize the pivot row so the pivot becomes 1.
		inv := gfInv(aug[col][col])
		for j := 0; j < 2*n; j++ {
			aug[col][j] = gfMul(aug[col][j], inv)
		}
		// Eliminate the column from every other row.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := aug[r][col]
			if f == 0 {
				continue
			}
			for j := 0; j < 2*n; j++ {
				aug[r][j] ^= gfMul(f, aug[col][j])
			}
		}
	}
	inv := make([][]byte, n)
	for i := range inv {
		inv[i] = make([]byte, n)
		copy(inv[i], aug[i][n:])
	}
	return inv, nil
}

// buildGenerator builds the n x k systematic generator matrix G.
//
// V is the n x k Vandermonde matrix with V[i][j] = (i+1)^j (the field
// element i+1 raised to the j-th power, 0-based rows). G = V * T^{-1}
// where T is the top k x k square of V, so the first k rows of G are the
// identity matrix and the remaining m rows produce parity shards.
func buildGenerator(k, m int) [][]byte {
	n := k + m
	v := make([][]byte, n)
	for i := 0; i < n; i++ {
		v[i] = make([]byte, k)
		for j := 0; j < k; j++ {
			v[i][j] = gfPow(byte(i+1), j)
		}
	}
	topInv, err := matInvert(v[:k])
	if err != nil {
		// Rows 1..k of a Vandermonde matrix over distinct nonzero
		// elements are always invertible; this cannot happen.
		panic("erasure: Vandermonde top is singular: " + err.Error())
	}
	return matMul(v, topInv)
}
