package erasure

import "testing"

// refMul is an independent bitwise (shift-and-reduce) reference
// implementation of GF(256) multiplication modulo 0x11d, used to
// cross-check the log/exp table implementation.
func refMul(a, b byte) byte {
	x := int(a)
	y := int(b)
	res := 0
	for y != 0 {
		if y&1 != 0 {
			res ^= x
		}
		y >>= 1
		x <<= 1
		if x&0x100 != 0 {
			x ^= fieldPoly
		}
	}
	return byte(res)
}

func TestMulMatchesBitwiseReference(t *testing.T) {
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b++ {
			got := gfMul(byte(a), byte(b))
			want := refMul(byte(a), byte(b))
			if got != want {
				t.Fatalf("gfMul(%d,%d)=%d, reference=%d", a, b, got, want)
			}
		}
	}
}

func TestFieldLaws(t *testing.T) {
	// 0x02 must generate the full multiplicative group (255 elements).
	seen := make(map[byte]bool)
	x := byte(1)
	for i := 0; i < 255; i++ {
		if seen[x] {
			t.Fatalf("generator cycle repeats after %d steps at %d", i, x)
		}
		seen[x] = true
		x = gfMul(x, 2)
	}
	if x != 1 {
		t.Fatalf("generator order is not 255: ended at %d", x)
	}
	if len(seen) != 255 {
		t.Fatalf("generator produced %d distinct elements, want 255", len(seen))
	}

	for a := 1; a < 256; a++ {
		if gfMul(byte(a), gfInv(byte(a))) != 1 {
			t.Fatalf("inv(%d) is not an inverse", a)
		}
	}
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b += 17 {
			if gfMul(byte(a), byte(b)) != gfMul(byte(b), byte(a)) {
				t.Fatalf("multiplication not commutative at %d,%d", a, b)
			}
		}
	}
}

func TestPow(t *testing.T) {
	for a := 0; a < 256; a++ {
		acc := byte(1)
		for e := 0; e < 10; e++ {
			if gfPow(byte(a), e) != acc {
				t.Fatalf("gfPow(%d,%d) mismatch", a, e)
			}
			acc = gfMul(acc, byte(a))
		}
	}
}
