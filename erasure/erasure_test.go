package erasure

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"
)

func randBytes(r *rand.Rand, n int) []byte {
	b := make([]byte, n)
	r.Read(b)
	return b
}

// pick returns the shards at the given indices as a Shard list.
func pick(shards [][]byte, idxs ...int) []Shard {
	out := make([]Shard, len(idxs))
	for i, idx := range idxs {
		out[i] = Shard{Index: idx, Data: shards[idx]}
	}
	return out
}

func combinations(n, k int) [][]int {
	var out [][]int
	var cur []int
	var rec func(start int)
	rec = func(start int) {
		if len(cur) == k {
			c := make([]int, k)
			copy(c, cur)
			out = append(out, c)
			return
		}
		for i := start; i <= n-(k-len(cur)); i++ {
			cur = append(cur, i)
			rec(i + 1)
			cur = cur[:len(cur)-1]
		}
	}
	rec(0)
	return out
}

func TestEncodeSystematic(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	data := randBytes(r, 100)
	k, m := 4, 2
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	if len(shards) != k+m {
		t.Fatalf("got %d shards, want %d", len(shards), k+m)
	}
	s := shardLen(len(data), k)
	for i, sh := range shards {
		if len(sh) != s {
			t.Fatalf("shard %d has length %d, want %d", i, len(sh), s)
		}
	}
	// First k shards are the data itself, contiguous, tail zero-padded.
	flat := bytes.Join(shards[:k], nil)
	if !bytes.Equal(flat[:len(data)], data) {
		t.Fatal("systematic data shards do not reproduce the input")
	}
	if !bytes.Equal(flat[len(data):], make([]byte, k*s-len(data))) {
		t.Fatal("tail padding is not zero")
	}
}

func TestGeneratorIdentityTop(t *testing.T) {
	for k := 1; k <= 16; k++ {
		for m := 1; m <= 8; m++ {
			g := buildGenerator(k, m)
			if len(g) != k+m {
				t.Fatalf("k=%d m=%d: %d rows", k, m, len(g))
			}
			for i := 0; i < k; i++ {
				for j := 0; j < k; j++ {
					want := byte(0)
					if i == j {
						want = 1
					}
					if g[i][j] != want {
						t.Fatalf("k=%d m=%d: G[%d][%d]=%d, want %d", k, m, i, j, g[i][j], want)
					}
				}
			}
		}
	}
}

func TestEmptyInput(t *testing.T) {
	k, m := 3, 2
	shards, err := Encode(nil, k, m)
	if err != nil {
		t.Fatal(err)
	}
	if len(shards) != k+m {
		t.Fatalf("got %d shards, want %d", len(shards), k+m)
	}
	for i, sh := range shards {
		if len(sh) != 0 {
			t.Fatalf("shard %d has length %d, want 0", i, len(sh))
		}
	}
	// Zero-length shards are valid; any k of them reconstruct the empty input.
	all, data, err := Reconstruct(0, pick(shards, 0, 2, 4), k, m)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 0 {
		t.Fatalf("recovered %d bytes, want 0", len(data))
	}
	if len(all) != k+m {
		t.Fatalf("got %d shards, want %d", len(all), k+m)
	}
	for i, sh := range all {
		if len(sh) != 0 {
			t.Fatalf("rebuilt shard %d has length %d, want 0", i, len(sh))
		}
	}
}

// TestExhaustiveSmallKM exhausts every size-k subset of available shards
// for all k<=3, m<=3, on fixed-seed samples of non-divisible lengths.
func TestExhaustiveSmallKM(t *testing.T) {
	r := rand.New(rand.NewSource(20260919))
	for k := 1; k <= 3; k++ {
		for m := 1; m <= 3; m++ {
			n := k + m
			// Non-divisible lengths (and one empty case for k=1).
			for _, size := range []int{1, k*3 + 1, 37} {
				data := randBytes(r, size)
				shards, err := Encode(data, k, m)
				if err != nil {
					t.Fatal(err)
				}
				for _, combo := range combinations(n, k) {
					all, got, err := Reconstruct(len(data), pick(shards, combo...), k, m)
					if err != nil {
						t.Fatalf("k=%d m=%d size=%d combo=%v: %v", k, m, size, combo, err)
					}
					if !bytes.Equal(got, data) {
						t.Fatalf("k=%d m=%d size=%d combo=%v: data mismatch", k, m, size, combo)
					}
					for i := 0; i < n; i++ {
						if !bytes.Equal(all[i], shards[i]) {
							t.Fatalf("k=%d m=%d size=%d combo=%v: shard %d mismatch", k, m, size, combo, i)
						}
					}
				}
			}
		}
	}
}

// TestRandomRoundTrip checks fixed-seed random samples across the full
// parameter range, dropping random subsets including data and parity shards.
func TestRandomRoundTrip(t *testing.T) {
	r := rand.New(rand.NewSource(42))
	for trial := 0; trial < 40; trial++ {
		k := 1 + r.Intn(16)
		m := 1 + r.Intn(8)
		n := k + m
		size := r.Intn(4096)
		data := randBytes(r, size)
		shards, err := Encode(data, k, m)
		if err != nil {
			t.Fatal(err)
		}
		// Keep a random subset of at least k shards.
		perm := r.Perm(n)
		keep := k
		if extra := n - k; extra > 0 {
			keep += r.Intn(extra + 1)
		}
		var avail []Shard
		for _, idx := range perm[:keep] {
			avail = append(avail, Shard{Index: idx, Data: shards[idx]})
		}
		all, got, err := Reconstruct(size, avail, k, m)
		if err != nil {
			t.Fatalf("trial %d (k=%d m=%d size=%d keep=%d): %v", trial, k, m, size, keep, err)
		}
		if !bytes.Equal(got, data) {
			t.Fatalf("trial %d: data mismatch", trial)
		}
		for i := 0; i < n; i++ {
			if !bytes.Equal(all[i], shards[i]) {
				t.Fatalf("trial %d: shard %d mismatch", trial, i)
			}
		}
	}
}

func TestMissingDataAndParityShards(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	data := randBytes(r, 1000) // non-divisible by k=6
	k, m := 6, 3
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	// Lose two data shards and one parity shard (the maximum tolerable).
	avail := pick(shards, 0, 2, 4, 5, 7, 8)
	_, got, err := Reconstruct(len(data), avail, k, m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("data mismatch after losing data and parity shards")
	}
	// Lose only parity shards.
	_, got, err = Reconstruct(len(data), pick(shards, 0, 1, 2, 3, 4, 5), k, m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("data mismatch after losing only parity shards")
	}
}

func TestRejects(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	k, m := 4, 2
	n := k + m
	data := randBytes(r, 100)
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	s := shardLen(len(data), k)

	if _, err := Encode(data, 0, m); !errors.Is(err, ErrBadK) {
		t.Fatalf("k=0: %v", err)
	}
	if _, err := Encode(data, 17, m); !errors.Is(err, ErrBadK) {
		t.Fatalf("k=17: %v", err)
	}
	if _, err := Encode(data, k, 0); !errors.Is(err, ErrBadM) {
		t.Fatalf("m=0: %v", err)
	}
	if _, err := Encode(data, k, 9); !errors.Is(err, ErrBadM) {
		t.Fatalf("m=9: %v", err)
	}
	if _, err := Encode(make([]byte, MaxInputSize+1), k, m); !errors.Is(err, ErrInputTooLarge) {
		t.Fatalf("oversize input: %v", err)
	}

	good := pick(shards, 0, 1, 2, 3)

	// Negative and oversized original length.
	if _, _, err := Reconstruct(-1, good, k, m); !errors.Is(err, ErrBadOrigLen) {
		t.Fatalf("negative origLen: %v", err)
	}
	if _, _, err := Reconstruct(MaxInputSize+1, good, k, m); !errors.Is(err, ErrBadOrigLen) {
		t.Fatalf("oversize origLen: %v", err)
	}
	// Wrong shard length.
	bad := pick(shards, 0, 1, 2, 3)
	bad[1].Data = make([]byte, s+1)
	if _, _, err := Reconstruct(len(data), bad, k, m); !errors.Is(err, ErrBadShardLen) {
		t.Fatalf("wrong shard length: %v", err)
	}
	// Wrong origLen changes s, so previously valid shards are rejected.
	if _, _, err := Reconstruct(len(data)+1, good, k, m); !errors.Is(err, ErrBadShardLen) {
		t.Fatalf("origLen off by one: %v", err)
	}
	// Fewer than k shards.
	if _, _, err := Reconstruct(len(data), good[:k-1], k, m); !errors.Is(err, ErrTooFewShards) {
		t.Fatalf("too few shards: %v", err)
	}
	// Duplicate index.
	dup := append(pick(shards, 0, 1, 2), Shard{Index: 2, Data: shards[2]})
	if _, _, err := Reconstruct(len(data), dup, k, m); !errors.Is(err, ErrDupIndex) {
		t.Fatalf("duplicate index: %v", err)
	}
	// Out-of-range index.
	oob := append(pick(shards, 0, 1, 2), Shard{Index: n, Data: shards[0]})
	if _, _, err := Reconstruct(len(data), oob, k, m); !errors.Is(err, ErrBadIndex) {
		t.Fatalf("out-of-range index: %v", err)
	}
	neg := append(pick(shards, 0, 1, 2), Shard{Index: -1, Data: shards[0]})
	if _, _, err := Reconstruct(len(data), neg, k, m); !errors.Is(err, ErrBadIndex) {
		t.Fatalf("negative index: %v", err)
	}
}

// TestExtraShardInconsistency corrupts one shard while providing more than
// k shards, which must be detected.
func TestExtraShardInconsistency(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	k, m := 3, 2
	data := randBytes(r, 91)
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	avail := pick(shards, 0, 1, 2, 3) // k+1 shards
	corrupt := make([]byte, len(avail[3].Data))
	copy(corrupt, avail[3].Data)
	corrupt[0] ^= 0xff
	avail[3].Data = corrupt
	if _, _, err := Reconstruct(len(data), avail, k, m); !errors.Is(err, ErrInconsistent) {
		t.Fatalf("corrupted extra shard: %v", err)
	}
}

// TestBadPaddingDetected supplies shards whose reconstruction would have
// nonzero tail padding (here: shards encoded from a longer input presented
// with a shorter origLen that happens to share the same shard length).
func TestBadPaddingDetected(t *testing.T) {
	r := rand.New(rand.NewSource(13))
	k, m := 4, 2
	// Same shard length s=2 for N=8 and N=7.
	data := randBytes(r, 8)
	for data[7] == 0 {
		data = randBytes(r, 8)
	}
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	// Claim origLen=7: shard length is still 2, but the recovered tail
	// byte is nonzero.
	if _, _, err := Reconstruct(7, pick(shards, 0, 1, 2, 3), k, m); !errors.Is(err, ErrBadPadding) {
		t.Fatalf("nonzero tail padding: %v", err)
	}
}

// TestInputsUnchanged verifies that Encode and Reconstruct never modify
// their inputs, including on error paths, and that outputs do not alias
// the inputs.
func TestInputsUnchanged(t *testing.T) {
	r := rand.New(rand.NewSource(17))
	k, m := 4, 3
	data := randBytes(r, 100)
	dataCopy := bytes.Clone(data)

	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, dataCopy) {
		t.Fatal("Encode modified its input")
	}
	// Mutating every returned shard must not affect the input.
	for _, sh := range shards {
		for i := range sh {
			sh[i] ^= 0xff
		}
	}
	if !bytes.Equal(data, dataCopy) {
		t.Fatal("Encode output aliases its input")
	}

	shards, err = Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	avail := pick(shards, 0, 2, 3, 5, 6)
	availCopy := make([]Shard, len(avail))
	for i, sh := range avail {
		availCopy[i] = Shard{Index: sh.Index, Data: bytes.Clone(sh.Data)}
	}

	all, got, err := Reconstruct(len(data), avail, k, m)
	if err != nil {
		t.Fatal(err)
	}
	for i, sh := range avail {
		if !bytes.Equal(sh.Data, availCopy[i].Data) {
			t.Fatalf("Reconstruct modified input shard %d", sh.Index)
		}
	}
	// Mutating outputs must not affect the provided shards.
	for _, sh := range all {
		for i := range sh {
			sh[i] ^= 0xff
		}
	}
	for i := range got {
		got[i] ^= 0xff
	}
	for i, sh := range avail {
		if !bytes.Equal(sh.Data, availCopy[i].Data) {
			t.Fatalf("Reconstruct output aliases input shard %d", sh.Index)
		}
	}

	// Error path: inputs still unchanged.
	bad := pick(shards, 0, 1, 2, 3)
	badCopy := make([]Shard, len(bad))
	for i, sh := range bad {
		badCopy[i] = Shard{Index: sh.Index, Data: bytes.Clone(sh.Data)}
	}
	if _, _, err := Reconstruct(-5, bad, k, m); err == nil {
		t.Fatal("expected error")
	}
	for i, sh := range bad {
		if !bytes.Equal(sh.Data, badCopy[i].Data) {
			t.Fatalf("error path modified input shard %d", sh.Index)
		}
	}
}
