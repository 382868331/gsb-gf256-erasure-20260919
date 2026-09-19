package erasure

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"
)

// gfMulRef is an independent bit-by-bit reference implementation of
// GF(256) multiplication (Russian-peasant style, polynomial 0x11d),
// used to cross-check the table-based gfMul.
func gfMulRef(a, b byte) byte {
	aa := int(a)
	bb := int(b)
	p := 0
	for i := 0; i < 8; i++ {
		if bb&1 != 0 {
			p ^= aa
		}
		carry := aa & 0x80
		aa <<= 1
		if carry != 0 {
			aa ^= 0x11d
		}
		bb >>= 1
	}
	return byte(p)
}

func TestGFMulAgainstBitwiseReference(t *testing.T) {
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b++ {
			got := gfMul(byte(a), byte(b))
			want := gfMulRef(byte(a), byte(b))
			if got != want {
				t.Fatalf("gfMul(%d,%d)=%d, reference=%d", a, b, got, want)
			}
			if gfMulTab[a][b] != want {
				t.Fatalf("gfMulTab[%d][%d]=%d, reference=%d", a, b, gfMulTab[a][b], want)
			}
		}
	}
}

func TestGFProperties(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 10000; i++ {
		a := byte(rng.Intn(256))
		b := byte(rng.Intn(256))
		c := byte(rng.Intn(256))
		if gfMul(a, b) != gfMul(b, a) {
			t.Fatal("multiplication not commutative")
		}
		if gfMul(gfMul(a, b), c) != gfMul(a, gfMul(b, c)) {
			t.Fatal("multiplication not associative")
		}
		if gfMul(a, b^c) != gfMul(a, b)^gfMul(a, c) {
			t.Fatal("multiplication not distributive over XOR")
		}
		if a != 0 && gfMul(a, gfInv(a)) != 1 {
			t.Fatal("inverse incorrect")
		}
	}
}

func TestGenMatrixSystematic(t *testing.T) {
	for k := 1; k <= MaxK; k++ {
		for m := 1; m <= MaxM; m++ {
			g := genMatrix(k, m)
			if len(g) != k+m {
				t.Fatalf("k=%d m=%d: got %d rows", k, m, len(g))
			}
			for i := 0; i < k; i++ {
				for j := 0; j < k; j++ {
					want := byte(0)
					if i == j {
						want = 1
					}
					if g[i][j] != want {
						t.Fatalf("k=%d m=%d: G[%d][%d]=%d, want identity", k, m, i, j, g[i][j])
					}
				}
			}
		}
	}
}

func TestEmptyInput(t *testing.T) {
	for _, data := range [][]byte{nil, {}} {
		shards, err := Encode(data, 3, 2)
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		if len(shards) != 5 {
			t.Fatalf("got %d shards, want 5", len(shards))
		}
		for i, sh := range shards {
			if len(sh) != 0 {
				t.Fatalf("shard %d has %d bytes, want 0", i, len(sh))
			}
		}
		// Zero-length shards are valid; reconstruct from k=3 of them.
		avail := []Shard{
			{Index: 0, Data: nil},
			{Index: 2, Data: []byte{}},
			{Index: 4, Data: shards[4]},
		}
		all, data, err := Reconstruct(avail, 3, 2, 0)
		if err != nil {
			t.Fatalf("Reconstruct: %v", err)
		}
		if len(all) != 5 || len(data) != 0 {
			t.Fatalf("got %d shards and %d data bytes", len(all), len(data))
		}
	}
}

func TestRoundTripNonDivisibleFixedSeed(t *testing.T) {
	rng := rand.New(rand.NewSource(20260919))
	for _, km := range [][2]int{{2, 1}, {3, 2}, {4, 3}, {5, 2}, {7, 4}, {16, 8}} {
		k, m := km[0], km[1]
		// Lengths deliberately not divisible by k.
		for _, n := range []int{1, k - 1, k + 1, 3*k + 2, 1000*k + 7} {
			if n < 1 {
				n = 1
			}
			data := make([]byte, n)
			rng.Read(data)
			shards, err := Encode(data, k, m)
			if err != nil {
				t.Fatalf("k=%d m=%d n=%d: Encode: %v", k, m, n, err)
			}
			s := shardLen(n, k)
			for i, sh := range shards {
				if len(sh) != s {
					t.Fatalf("k=%d n=%d: shard %d len %d, want %d", k, n, i, len(sh), s)
				}
			}
			// Drop m random shards.
			perm := rng.Perm(k + m)
			drop := make(map[int]bool)
			for _, idx := range perm[:m] {
				drop[idx] = true
			}
			var avail []Shard
			for i, sh := range shards {
				if !drop[i] {
					avail = append(avail, Shard{Index: i, Data: sh})
				}
			}
			all, got, err := Reconstruct(avail, k, m, n)
			if err != nil {
				t.Fatalf("k=%d m=%d n=%d: Reconstruct: %v", k, m, n, err)
			}
			if !bytes.Equal(got, data) {
				t.Fatalf("k=%d m=%d n=%d: data mismatch", k, m, n)
			}
			for i := range all {
				if !bytes.Equal(all[i], shards[i]) {
					t.Fatalf("k=%d m=%d n=%d: shard %d mismatch", k, m, n, i)
				}
			}
		}
	}
}

// TestExhaustiveSmallK exhaustively checks every size-k subset of
// available indices for all k<=3, m<=3.
func TestExhaustiveSmallK(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for k := 1; k <= 3; k++ {
		for m := 1; m <= 3; m++ {
			n := k + m
			data := make([]byte, 5*k+1) // not divisible by k
			rng.Read(data)
			shards, err := Encode(data, k, m)
			if err != nil {
				t.Fatalf("k=%d m=%d: Encode: %v", k, m, err)
			}
			var combos [][]int
			var pick func(start int, cur []int)
			pick = func(start int, cur []int) {
				if len(cur) == k {
					c := make([]int, k)
					copy(c, cur)
					combos = append(combos, c)
					return
				}
				for i := start; i < n; i++ {
					pick(i+1, append(cur, i))
				}
			}
			pick(0, nil)
			for _, combo := range combos {
				avail := make([]Shard, k)
				for i, idx := range combo {
					avail[i] = Shard{Index: idx, Data: shards[idx]}
				}
				all, got, err := Reconstruct(avail, k, m, len(data))
				if err != nil {
					t.Fatalf("k=%d m=%d combo=%v: %v", k, m, combo, err)
				}
				if !bytes.Equal(got, data) {
					t.Fatalf("k=%d m=%d combo=%v: data mismatch", k, m, combo)
				}
				for i := range all {
					if !bytes.Equal(all[i], shards[i]) {
						t.Fatalf("k=%d m=%d combo=%v: shard %d mismatch", k, m, combo, i)
					}
				}
			}
		}
	}
}

func TestDataAndParityShardsMissing(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	k, m := 6, 4
	data := make([]byte, 12345)
	rng.Read(data)
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	// Lose m pure data shards.
	var avail []Shard
	for i := m; i < k+m; i++ {
		avail = append(avail, Shard{Index: i, Data: shards[i]})
	}
	if _, got, err := Reconstruct(avail, k, m, len(data)); err != nil || !bytes.Equal(got, data) {
		t.Fatalf("data shards lost: err=%v match=%v", err, bytes.Equal(got, data))
	}
	// Lose all m parity shards.
	avail = nil
	for i := 0; i < k; i++ {
		avail = append(avail, Shard{Index: i, Data: shards[i]})
	}
	if _, got, err := Reconstruct(avail, k, m, len(data)); err != nil || !bytes.Equal(got, data) {
		t.Fatalf("parity shards lost: err=%v match=%v", err, bytes.Equal(got, data))
	}
	// Lose a mix.
	avail = nil
	for i := 0; i < k+m; i++ {
		if i == 1 || i == k || i == k+2 || i == 4 {
			continue
		}
		avail = append(avail, Shard{Index: i, Data: shards[i]})
	}
	if _, got, err := Reconstruct(avail, k, m, len(data)); err != nil || !bytes.Equal(got, data) {
		t.Fatalf("mixed loss: err=%v match=%v", err, bytes.Equal(got, data))
	}
}

func TestTooFewShards(t *testing.T) {
	data := []byte("hello world, hello erasure coding")
	shards, err := Encode(data, 4, 2)
	if err != nil {
		t.Fatal(err)
	}
	avail := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 1, Data: shards[1]},
		{Index: 5, Data: shards[5]},
	}
	if _, _, err := Reconstruct(avail, 4, 2, len(data)); !errors.Is(err, ErrTooFewShards) {
		t.Fatalf("got %v, want ErrTooFewShards", err)
	}
}

func TestBadOriginalLength(t *testing.T) {
	data := []byte("some data")
	shards, err := Encode(data, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	avail := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 1, Data: shards[1]},
	}
	if _, _, err := Reconstruct(avail, 2, 1, -1); !errors.Is(err, ErrNegativeLength) {
		t.Fatalf("negative: got %v, want ErrNegativeLength", err)
	}
	// origLen whose ceil(N/k) disagrees with the actual shard length.
	if _, _, err := Reconstruct(avail, 2, 1, len(data)+2); !errors.Is(err, ErrBadShardLen) {
		t.Fatalf("mismatched N: got %v, want ErrBadShardLen", err)
	}
	if _, _, err := Reconstruct(avail, 2, 1, MaxInputSize+1); !errors.Is(err, ErrInputTooLarge) {
		t.Fatalf("oversize: got %v, want ErrInputTooLarge", err)
	}
	if _, err := Encode(make([]byte, MaxInputSize+1), 2, 1); !errors.Is(err, ErrInputTooLarge) {
		t.Fatalf("encode oversize: got %v, want ErrInputTooLarge", err)
	}
}

func TestBadShardLength(t *testing.T) {
	data := []byte("some data")
	shards, err := Encode(data, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	bad := make([]byte, len(shards[0])+1)
	avail := []Shard{
		{Index: 0, Data: bad},
		{Index: 1, Data: shards[1]},
	}
	if _, _, err := Reconstruct(avail, 2, 1, len(data)); !errors.Is(err, ErrBadShardLen) {
		t.Fatalf("got %v, want ErrBadShardLen", err)
	}
}

func TestDuplicateAndOutOfRangeIndex(t *testing.T) {
	data := []byte("some data")
	shards, err := Encode(data, 2, 1)
	if err != nil {
		t.Fatal(err)
	}
	dup := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 0, Data: shards[0]},
	}
	if _, _, err := Reconstruct(dup, 2, 1, len(data)); !errors.Is(err, ErrDuplicateIndex) {
		t.Fatalf("got %v, want ErrDuplicateIndex", err)
	}
	oob := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 3, Data: shards[1]},
	}
	if _, _, err := Reconstruct(oob, 2, 1, len(data)); !errors.Is(err, ErrIndexOutOfRange) {
		t.Fatalf("got %v, want ErrIndexOutOfRange", err)
	}
	neg := []Shard{
		{Index: -1, Data: shards[0]},
		{Index: 1, Data: shards[1]},
	}
	if _, _, err := Reconstruct(neg, 2, 1, len(data)); !errors.Is(err, ErrIndexOutOfRange) {
		t.Fatalf("got %v, want ErrIndexOutOfRange", err)
	}
}

func TestInvalidKM(t *testing.T) {
	for _, km := range [][2]int{{0, 1}, {17, 1}, {1, 0}, {1, 9}, {-1, 1}} {
		if _, err := Encode([]byte("x"), km[0], km[1]); err == nil {
			t.Fatalf("k=%d m=%d: Encode accepted", km[0], km[1])
		}
		if _, _, err := Reconstruct(nil, km[0], km[1], 0); err == nil {
			t.Fatalf("k=%d m=%d: Reconstruct accepted", km[0], km[1])
		}
	}
}

// TestExtraShardInconsistent: with more than k shards, a corrupted shard
// is detected by re-checking every given shard.
func TestExtraShardInconsistent(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	data := make([]byte, 100)
	rng.Read(data)
	k, m := 2, 1
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	corrupt := make([]byte, len(shards[2]))
	copy(corrupt, shards[2])
	corrupt[0] ^= 0xff
	avail := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 1, Data: shards[1]},
		{Index: 2, Data: corrupt},
	}
	if _, _, err := Reconstruct(avail, k, m, len(data)); !errors.Is(err, ErrInconsistent) {
		t.Fatalf("got %v, want ErrInconsistent", err)
	}
}

// TestNonZeroPaddingDetected: with exactly k shards, non-zero tail
// padding in the recovered data is still detectable and rejected.
func TestNonZeroPaddingDetected(t *testing.T) {
	data := []byte{1, 2, 3} // k=2 -> s=2, shard1 = {3, 0}
	k, m := 2, 1
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	bad := make([]byte, len(shards[1]))
	copy(bad, shards[1])
	bad[len(bad)-1] = 0xaa // corrupt the padding byte
	avail := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 1, Data: bad},
	}
	if _, _, err := Reconstruct(avail, k, m, len(data)); !errors.Is(err, ErrBadPadding) {
		t.Fatalf("got %v, want ErrBadPadding", err)
	}
}

// TestInputsUnchanged verifies Encode and Reconstruct never modify their
// inputs and that outputs alias no input memory.
func TestInputsUnchanged(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	k, m := 4, 3
	data := make([]byte, 1000)
	rng.Read(data)
	dataCopy := append([]byte(nil), data...)

	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(data, dataCopy) {
		t.Fatal("Encode modified its input")
	}
	// Outputs must not alias the input: mutating data must not change shards.
	for i := range data {
		data[i] ^= 0xff
	}
	snapshot := make([][]byte, len(shards))
	for i, sh := range shards {
		snapshot[i] = append([]byte(nil), sh...)
	}
	copy(data, dataCopy)
	for i, sh := range shards {
		if !bytes.Equal(sh, snapshot[i]) {
			t.Fatalf("shard %d aliases the input data", i)
		}
	}

	avail := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 2, Data: shards[2]},
		{Index: 3, Data: shards[3]},
		{Index: 6, Data: shards[6]},
	}
	availCopy := make([][]byte, len(avail))
	for i, sh := range avail {
		availCopy[i] = append([]byte(nil), sh.Data...)
	}
	all, got, err := Reconstruct(avail, k, m, len(data))
	if err != nil {
		t.Fatal(err)
	}
	for i, sh := range avail {
		if !bytes.Equal(sh.Data, availCopy[i]) {
			t.Fatalf("Reconstruct modified input shard %d", i)
		}
	}
	// Mutating outputs must not affect the input shards.
	for _, sh := range all {
		for i := range sh {
			sh[i] ^= 0xff
		}
	}
	for i := range got {
		got[i] ^= 0xff
	}
	for i, sh := range avail {
		if !bytes.Equal(sh.Data, availCopy[i]) {
			t.Fatalf("output aliases input shard %d", i)
		}
	}
}

// TestMaxInputSize exercises the 4 MiB boundary once with minimal k, m.
func TestMaxInputSize(t *testing.T) {
	rng := rand.New(rand.NewSource(13))
	data := make([]byte, MaxInputSize)
	rng.Read(data)
	k, m := 4, 1
	shards, err := Encode(data, k, m)
	if err != nil {
		t.Fatal(err)
	}
	avail := []Shard{
		{Index: 0, Data: shards[0]},
		{Index: 2, Data: shards[2]},
		{Index: 3, Data: shards[3]},
		{Index: 4, Data: shards[4]},
	}
	_, got, err := Reconstruct(avail, k, m, len(data))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, data) {
		t.Fatal("4 MiB round trip mismatch")
	}
}
