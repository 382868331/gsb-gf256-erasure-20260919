package erasure

import "errors"

// errSingular is returned when a matrix has no inverse over GF(256).
var errSingular = errors.New("erasure: singular matrix")

// vandermonde returns the rows x cols Vandermonde matrix with
// V[i][j] = (i+1)^j (row i uses the field element i+1, 0-based rows).
func vandermonde(rows, cols int) [][]byte {
	m := make([][]byte, rows)
	for i := 0; i < rows; i++ {
		m[i] = make([]byte, cols)
		base := byte(i + 1)
		v := byte(1) // (i+1)^0
		for j := 0; j < cols; j++ {
			m[i][j] = v
			v = gfMul(v, base)
		}
	}
	return m
}

// matMul returns a*b over GF(256). a is r x c, b is c x n.
func matMul(a, b [][]byte) [][]byte {
	r := len(a)
	c := len(b)
	n := len(b[0])
	out := make([][]byte, r)
	for i := 0; i < r; i++ {
		out[i] = make([]byte, n)
		for t := 0; t < c; t++ {
			coef := a[i][t]
			if coef == 0 {
				continue
			}
			row := b[t]
			tab := gfMulTab[coef]
			for j := 0; j < n; j++ {
				out[i][j] ^= tab[row[j]]
			}
		}
	}
	return out
}

// matInv returns the inverse of the square matrix m over GF(256),
// computed by Gauss-Jordan elimination. m is not modified.
func matInv(m [][]byte) ([][]byte, error) {
	n := len(m)
	// Augment [m | I].
	a := make([][]byte, n)
	for i := 0; i < n; i++ {
		a[i] = make([]byte, 2*n)
		copy(a[i], m[i])
		a[i][n+i] = 1
	}
	for col := 0; col < n; col++ {
		pivot := -1
		for r := col; r < n; r++ {
			if a[r][col] != 0 {
				pivot = r
				break
			}
		}
		if pivot < 0 {
			return nil, errSingular
		}
		a[col], a[pivot] = a[pivot], a[col]
		// Normalize the pivot row.
		inv := gfInv(a[col][col])
		tab := gfMulTab[inv]
		for c := 0; c < 2*n; c++ {
			a[col][c] = tab[a[col][c]]
		}
		// Eliminate the column from every other row.
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col]
			if f == 0 {
				continue
			}
			ftab := gfMulTab[f]
			for c := 0; c < 2*n; c++ {
				a[r][c] ^= ftab[a[col][c]]
			}
		}
	}
	inv := make([][]byte, n)
	for i := 0; i < n; i++ {
		inv[i] = make([]byte, n)
		copy(inv[i], a[i][n:])
	}
	return inv, nil
}

// genMatrix builds the n x k systematic generator matrix
// G = V * inv(top k rows of V), whose first k rows are the identity.
func genMatrix(k, m int) [][]byte {
	n := k + m
	v := vandermonde(n, k)
	top := make([][]byte, k)
	for i := 0; i < k; i++ {
		top[i] = v[i]
	}
	topInv, err := matInv(top)
	if err != nil {
		// Rows 1..k of a Vandermonde matrix over distinct non-zero
		// elements are always invertible; this cannot happen.
		panic(err)
	}
	return matMul(v, topInv)
}
