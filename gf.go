// Package erasure implements a systematic erasure code over GF(256)
// (field polynomial 0x11d, addition is XOR) for recovering known-missing
// shards of a file split into k data shards plus m parity shards.
package erasure

// GF(256) arithmetic with the irreducible polynomial x^8+x^4+x^3+x+1
// (0x11d) and generator 0x02. Addition/subtraction are XOR.

const gfPoly = 0x11d

var (
	// gfExp[i] = 2^i for i in [0,255); doubled to 512 to avoid modular
	// reduction on index addition in gfMul.
	gfExp [512]byte
	// gfLog[a] = i such that 2^i == a; gfLog[0] is unused.
	gfLog [256]byte
	// gfMulTab[a][b] = a*b in GF(256); full table for fast encoding.
	gfMulTab [256][256]byte
)

func init() {
	x := 1
	for i := 0; i < 255; i++ {
		gfExp[i] = byte(x)
		gfLog[byte(x)] = byte(i)
		x <<= 1
		if x&0x100 != 0 {
			x ^= gfPoly
		}
	}
	for i := 255; i < 512; i++ {
		gfExp[i] = gfExp[i-255]
	}
	for a := 0; a < 256; a++ {
		for b := 0; b < 256; b++ {
			gfMulTab[a][b] = gfMul(byte(a), byte(b))
		}
	}
}

// gfMul returns a*b in GF(256).
func gfMul(a, b byte) byte {
	if a == 0 || b == 0 {
		return 0
	}
	return gfExp[int(gfLog[a])+int(gfLog[b])]
}

// gfInv returns the multiplicative inverse of a. a must be non-zero.
func gfInv(a byte) byte {
	return gfExp[255-int(gfLog[a])]
}
