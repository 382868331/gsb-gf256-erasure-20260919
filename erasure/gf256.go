package erasure

// GF(256) arithmetic modulo the irreducible polynomial x^8+x^4+x^3+x+1
// (0x11d). Addition is XOR. The element 0x02 is a generator of the
// multiplicative group, so log/exp tables are built from powers of 2.

const fieldPoly = 0x11d

var gfExp [512]byte // exp table, doubled to avoid modular reduction on add
var gfLog [256]byte

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		gfExp[i] = byte(x)
		gfLog[byte(x)] = byte(i)
		x <<= 1
		if x&0x100 != 0 {
			x ^= fieldPoly
		}
	}
	for i := 255; i < 512; i++ {
		gfExp[i] = gfExp[i-255]
	}
}

// gfMul returns a*b in GF(256).
func gfMul(a, b byte) byte {
	if a == 0 || b == 0 {
		return 0
	}
	return gfExp[int(gfLog[a])+int(gfLog[b])]
}

// gfInv returns the multiplicative inverse of a. It panics if a == 0.
func gfInv(a byte) byte {
	if a == 0 {
		panic("erasure: inverse of zero in GF(256)")
	}
	return gfExp[255-int(gfLog[a])]
}

// gfPow returns a^e in GF(256) for e >= 0. By convention 0^0 == 1.
func gfPow(a byte, e int) byte {
	if e < 0 {
		panic("erasure: negative exponent")
	}
	if e == 0 {
		return 1
	}
	if a == 0 {
		return 0
	}
	return gfExp[(int(gfLog[a])*e)%255]
}

// mulAdd XORs c*src into dst: dst[i] ^= c*src[i]. dst and src must have the
// same length and must not overlap in a way that aliases dst (callers use
// freshly allocated dst).
func mulAdd(dst, src []byte, c byte) {
	switch c {
	case 0:
		return
	case 1:
		for i, v := range src {
			dst[i] ^= v
		}
	default:
		for i, v := range src {
			if v != 0 {
				dst[i] ^= gfExp[int(gfLog[c])+int(gfLog[v])]
			}
		}
	}
}
