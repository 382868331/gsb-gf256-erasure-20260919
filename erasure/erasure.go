// Package erasure implements a systematic erasure code over GF(256) for
// offline recovery of files split into shards when some shards are known
// to be lost. It only recovers shards that are explicitly reported missing;
// it does not promise to correct silently corrupted data.
//
// The field uses the polynomial 0x11d with XOR addition. An n x k
// Vandermonde matrix V with V[i][j] = (i+1)^j is turned into a systematic
// generator G = V * T^{-1} (T = top k rows of V), so the first k shards
// are the data itself and the remaining m shards are parity.
//
// Limits: 1 <= k <= 16, 1 <= m <= 8, n = k+m, input at most 4 MiB.
package erasure

import (
	"bytes"
	"errors"
	"fmt"
	"sort"
)

// MaxInputSize is the largest accepted original input length (4 MiB).
const MaxInputSize = 4 << 20

// Parameter and input validation errors.
var (
	ErrBadK          = errors.New("erasure: k must satisfy 1 <= k <= 16")
	ErrBadM          = errors.New("erasure: m must satisfy 1 <= m <= 8")
	ErrInputTooLarge = errors.New("erasure: input exceeds 4 MiB")
	ErrBadOrigLen    = errors.New("erasure: original length out of range")
	ErrTooFewShards  = errors.New("erasure: fewer than k shards provided")
	ErrBadIndex      = errors.New("erasure: shard index out of range")
	ErrDupIndex      = errors.New("erasure: duplicate shard index")
	ErrBadShardLen   = errors.New("erasure: shard length does not equal ceil(N/k)")
	ErrInconsistent  = errors.New("erasure: provided shards are inconsistent")
	ErrBadPadding    = errors.New("erasure: recovered data has nonzero tail padding")
)

// Shard is one available shard of an encoded object. Index is the 0-based
// shard position in [0, n); Data must have length ceil(origLen/k). A
// zero-length Data is a valid shard when the original input is empty.
type Shard struct {
	Index int
	Data  []byte
}

func checkParams(k, m int) error {
	if k < 1 || k > 16 {
		return ErrBadK
	}
	if m < 1 || m > 8 {
		return ErrBadM
	}
	return nil
}

// shardLen returns the per-shard length s = ceil(N/k) for original length n.
// shardLen(0) == 0.
func shardLen(n, k int) int {
	return (n + k - 1) / k
}

// Encode splits data into n = k+m shards: k systematic data shards followed
// by m parity shards, each of length ceil(len(data)/k). The data is split
// contiguously and zero-padded at the tail of the last data shard. Empty
// input yields n valid zero-length shards.
//
// The returned shards are freshly allocated and never alias data.
func Encode(data []byte, k, m int) ([][]byte, error) {
	if err := checkParams(k, m); err != nil {
		return nil, err
	}
	if len(data) > MaxInputSize {
		return nil, ErrInputTooLarge
	}
	n := k + m
	s := shardLen(len(data), k)
	shards := make([][]byte, n)
	for i := range shards {
		shards[i] = make([]byte, s)
	}
	for i := 0; i < k; i++ {
		lo := i * s
		if lo >= len(data) {
			break
		}
		hi := lo + s
		if hi > len(data) {
			hi = len(data)
		}
		copy(shards[i], data[lo:hi])
	}
	g := buildGenerator(k, m)
	for i := k; i < n; i++ {
		for j := 0; j < k; j++ {
			mulAdd(shards[i], shards[j], g[i][j])
		}
	}
	return shards, nil
}

// Reconstruct rebuilds all n shards and the original data from any at least
// k available shards. origLen is the original input length N; every provided
// shard must have length ceil(N/k). Missing shards are simply absent from
// the list; indices are 0-based.
//
// It rejects duplicate or out-of-range indices, negative or oversized
// origLen, shard lengths other than ceil(N/k), and fewer than k shards.
// After solving it re-verifies every provided shard against the
// reconstruction and checks that the zero tail padding of the recovered
// data is intact; any mismatch is reported as an error. Note that with
// exactly k shards, inconsistency among them cannot in general be detected
// and is not promised to be caught.
//
// Inputs are never modified, including on error paths. The returned shards
// and data are freshly allocated and never alias the input shards.
func Reconstruct(origLen int, shards []Shard, k, m int) (all [][]byte, data []byte, err error) {
	if err := checkParams(k, m); err != nil {
		return nil, nil, err
	}
	if origLen < 0 || origLen > MaxInputSize {
		return nil, nil, ErrBadOrigLen
	}
	n := k + m
	s := shardLen(origLen, k)
	if len(shards) < k {
		return nil, nil, ErrTooFewShards
	}
	seen := make([]bool, n)
	for _, sh := range shards {
		if sh.Index < 0 || sh.Index >= n {
			return nil, nil, fmt.Errorf("%w: %d", ErrBadIndex, sh.Index)
		}
		if seen[sh.Index] {
			return nil, nil, fmt.Errorf("%w: %d", ErrDupIndex, sh.Index)
		}
		seen[sh.Index] = true
		if len(sh.Data) != s {
			return nil, nil, fmt.Errorf("%w: got %d, want %d", ErrBadShardLen, len(sh.Data), s)
		}
	}

	g := buildGenerator(k, m)

	// Pick the k lowest-index available shards and solve for the data.
	picked := make([]Shard, k)
	{
		sorted := make([]Shard, len(shards))
		copy(sorted, shards)
		sort.Slice(sorted, func(a, b int) bool { return sorted[a].Index < sorted[b].Index })
		copy(picked, sorted[:k])
	}
	sub := make([][]byte, k)
	for i, sh := range picked {
		sub[i] = g[sh.Index]
	}
	subInv, err := matInvert(sub)
	if err != nil {
		return nil, nil, err
	}
	dataShards := make([][]byte, k)
	for i := 0; i < k; i++ {
		dataShards[i] = make([]byte, s)
		for j := 0; j < k; j++ {
			mulAdd(dataShards[i], picked[j].Data, subInv[i][j])
		}
	}

	// Recompute every shard from the recovered data shards.
	all = make([][]byte, n)
	for i := 0; i < n; i++ {
		all[i] = make([]byte, s)
		for j := 0; j < k; j++ {
			mulAdd(all[i], dataShards[j], g[i][j])
		}
	}

	// Re-verify every provided shard against the reconstruction.
	for _, sh := range shards {
		if !bytes.Equal(all[sh.Index], sh.Data) {
			return nil, nil, fmt.Errorf("%w: shard %d", ErrInconsistent, sh.Index)
		}
	}

	// Check the zero tail padding beyond the original length.
	for i := 0; i < k; i++ {
		ds := dataShards[i]
		lo := i * s
		for p := 0; p < s; p++ {
			if lo+p >= origLen && ds[p] != 0 {
				return nil, nil, ErrBadPadding
			}
		}
	}

	data = make([]byte, origLen)
	for i := 0; i < k; i++ {
		lo := i * s
		if lo >= origLen {
			break
		}
		hi := lo + s
		if hi > origLen {
			hi = origLen
		}
		copy(data[lo:hi], dataShards[i][:hi-lo])
	}
	return all, data, nil
}
