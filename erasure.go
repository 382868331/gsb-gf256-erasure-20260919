package erasure

import (
	"errors"
	"fmt"
)

// Limits of the codec.
const (
	// MaxK is the maximum number of data shards.
	MaxK = 16
	// MaxM is the maximum number of parity shards.
	MaxM = 8
	// MaxInputSize is the maximum input length in bytes (4 MiB).
	MaxInputSize = 4 << 20
)

// Errors returned by Encode and Reconstruct. Use errors.Is to match.
var (
	ErrInvalidK        = errors.New("erasure: k out of range [1,16]")
	ErrInvalidM        = errors.New("erasure: m out of range [1,8]")
	ErrInputTooLarge   = errors.New("erasure: input exceeds 4 MiB")
	ErrNegativeLength  = errors.New("erasure: negative original length")
	ErrTooFewShards    = errors.New("erasure: fewer than k shards given")
	ErrDuplicateIndex  = errors.New("erasure: duplicate shard index")
	ErrIndexOutOfRange = errors.New("erasure: shard index out of range")
	ErrBadShardLen     = errors.New("erasure: shard length does not equal ceil(N/k)")
	ErrBadPadding      = errors.New("erasure: non-zero tail padding in recovered data")
	ErrInconsistent    = errors.New("erasure: given shard inconsistent with reconstruction")
)

// Shard is one available shard of an encoded object, identified by its
// 0-based index in [0, k+m). Missing shards simply do not appear in the
// list given to Reconstruct. A zero-length Data is a valid shard when
// the original length N is 0 (every shard then has length 0).
type Shard struct {
	Index int
	Data  []byte
}

// shardLen returns s = ceil(N/k), the length of every shard.
func shardLen(n, k int) int {
	return (n + k - 1) / k
}

func checkParams(k, m int) error {
	if k < 1 || k > MaxK {
		return fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if m < 1 || m > MaxM {
		return fmt.Errorf("%w: m=%d", ErrInvalidM, m)
	}
	return nil
}

// Encode splits data into k data shards of length s = ceil(len(data)/k)
// (zero-padded at the tail of the last shard) and computes m parity
// shards, returning all n = k+m shards in index order. An empty input
// yields n empty shards (s = 0). The returned shards are freshly
// allocated and share no memory with data.
func Encode(data []byte, k, m int) ([][]byte, error) {
	if err := checkParams(k, m); err != nil {
		return nil, err
	}
	if len(data) > MaxInputSize {
		return nil, fmt.Errorf("%w: %d > %d", ErrInputTooLarge, len(data), MaxInputSize)
	}
	n := k + m
	s := shardLen(len(data), k)
	g := genMatrix(k, m)

	// Zero-padded data shards (internal working copies).
	padded := make([][]byte, k)
	for j := 0; j < k; j++ {
		p := make([]byte, s)
		start := j * s
		if start < len(data) {
			end := start + s
			if end > len(data) {
				end = len(data)
			}
			copy(p, data[start:end])
		}
		padded[j] = p
	}

	shards := make([][]byte, n)
	for i := 0; i < n; i++ {
		out := make([]byte, s)
		row := g[i]
		for j := 0; j < k; j++ {
			coef := row[j]
			if coef == 0 {
				continue
			}
			tab := gfMulTab[coef]
			src := padded[j]
			for x := 0; x < s; x++ {
				out[x] ^= tab[src[x]]
			}
		}
		shards[i] = out
	}
	return shards, nil
}

// Reconstruct rebuilds all n shards and the original data from any at
// least k available shards. origLen is the original data length N; every
// given shard must have length ceil(N/k). After decoding, the zero tail
// padding of the recovered data is verified and every given shard is
// re-checked against the reconstruction; any mismatch is an error.
//
// Note: with exactly k shards, an inconsistent set cannot in general be
// detected; no such guarantee is made.
//
// On error the inputs are left unmodified and nil results are returned.
// The returned slices are freshly allocated and alias no input memory.
func Reconstruct(shards []Shard, k, m, origLen int) (all [][]byte, data []byte, err error) {
	if err := checkParams(k, m); err != nil {
		return nil, nil, err
	}
	if origLen < 0 {
		return nil, nil, fmt.Errorf("%w: %d", ErrNegativeLength, origLen)
	}
	if origLen > MaxInputSize {
		return nil, nil, fmt.Errorf("%w: %d > %d", ErrInputTooLarge, origLen, MaxInputSize)
	}
	n := k + m
	s := shardLen(origLen, k)
	if len(shards) < k {
		return nil, nil, fmt.Errorf("%w: got %d, need %d", ErrTooFewShards, len(shards), k)
	}
	seen := make([]bool, n)
	for _, sh := range shards {
		if sh.Index < 0 || sh.Index >= n {
			return nil, nil, fmt.Errorf("%w: %d not in [0,%d)", ErrIndexOutOfRange, sh.Index, n)
		}
		if seen[sh.Index] {
			return nil, nil, fmt.Errorf("%w: %d", ErrDuplicateIndex, sh.Index)
		}
		seen[sh.Index] = true
		if len(sh.Data) != s {
			return nil, nil, fmt.Errorf("%w: shard %d has %d bytes, want %d", ErrBadShardLen, sh.Index, len(sh.Data), s)
		}
	}

	g := genMatrix(k, m)

	// Decode the k data shards from the first k available shards.
	sub := shards[:k]
	a := make([][]byte, k)
	for i, sh := range sub {
		a[i] = g[sh.Index]
	}
	aInv, err := matInv(a)
	if err != nil {
		return nil, nil, err
	}
	decoded := make([][]byte, k)
	for j := 0; j < k; j++ {
		out := make([]byte, s)
		row := aInv[j]
		for i := 0; i < k; i++ {
			coef := row[i]
			if coef == 0 {
				continue
			}
			tab := gfMulTab[coef]
			src := sub[i].Data
			for x := 0; x < s; x++ {
				out[x] ^= tab[src[x]]
			}
		}
		decoded[j] = out
	}

	// Verify the zero tail padding of the recovered data.
	if s > 0 {
		paddedLen := k * s
		for pos := origLen; pos < paddedLen; pos++ {
			if decoded[pos/s][pos%s] != 0 {
				return nil, nil, fmt.Errorf("%w: byte %d of padded data", ErrBadPadding, pos)
			}
		}
	}

	// Rebuild all n shards from the decoded data.
	all = make([][]byte, n)
	for i := 0; i < n; i++ {
		out := make([]byte, s)
		row := g[i]
		for j := 0; j < k; j++ {
			coef := row[j]
			if coef == 0 {
				continue
			}
			tab := gfMulTab[coef]
			src := decoded[j]
			for x := 0; x < s; x++ {
				out[x] ^= tab[src[x]]
			}
		}
		all[i] = out
	}

	// Re-check every given shard against the reconstruction.
	for _, sh := range shards {
		want := all[sh.Index]
		got := sh.Data
		for x := 0; x < s; x++ {
			if got[x] != want[x] {
				return nil, nil, fmt.Errorf("%w: shard %d differs at byte %d", ErrInconsistent, sh.Index, x)
			}
		}
	}

	data = make([]byte, origLen)
	for j := 0; j < k && j*s < origLen; j++ {
		end := (j + 1) * s
		if end > origLen {
			end = origLen
		}
		copy(data[j*s:end], decoded[j][:end-j*s])
	}
	return all, data, nil
}
