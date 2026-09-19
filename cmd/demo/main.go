// Command demo shows a successful recovery from a double shard loss and
// a real failure when fewer than k shards are available.
package main

import (
	"bytes"
	"fmt"
	"math/rand"
	"os"
	"time"

	erasure "github.com/382868331/gsb-gf256-erasure-20260919"
)

func main() {
	start := time.Now()
	k, m := 6, 3
	n := k + m

	rng := rand.New(rand.NewSource(20260919))
	data := make([]byte, 100_003) // deliberately not divisible by k
	rng.Read(data)

	shards, err := erasure.Encode(data, k, m)
	if err != nil {
		fmt.Println("encode failed:", err)
		os.Exit(1)
	}
	fmt.Printf("encoded %d bytes into n=%d shards (k=%d data + m=%d parity), shard size s=%d\n",
		len(data), n, k, m, len(shards[0]))

	// --- Normal case: lose two shards (one data, one parity), recover. ---
	lost := []int{2, 7}
	var avail []erasure.Shard
	for i := 0; i < n; i++ {
		if i == lost[0] || i == lost[1] {
			continue
		}
		avail = append(avail, erasure.Shard{Index: i, Data: shards[i]})
	}
	fmt.Printf("lost shards %v (data shard %d, parity shard %d); %d shards remain\n",
		lost, lost[0], lost[1], len(avail))

	all, recovered, err := erasure.Reconstruct(avail, k, m, len(data))
	if err != nil {
		fmt.Println("reconstruct failed:", err)
		os.Exit(1)
	}
	if !bytes.Equal(recovered, data) {
		fmt.Println("FAIL: recovered data does not match the original")
		os.Exit(1)
	}
	for i := 0; i < n; i++ {
		if !bytes.Equal(all[i], shards[i]) {
			fmt.Printf("FAIL: rebuilt shard %d does not match\n", i)
			os.Exit(1)
		}
	}
	fmt.Printf("OK: recovered all %d shards and the original %d bytes; "+
		"rebuilt shards %v match the originals\n", n, len(recovered), lost)

	// --- Failure case: only k-1 shards available, recovery must fail. ---
	short := avail[:k-1]
	_, _, err = erasure.Reconstruct(short, k, m, len(data))
	if err == nil {
		fmt.Println("FAIL: reconstruct with k-1 shards unexpectedly succeeded")
		os.Exit(1)
	}
	fmt.Printf("OK: reconstruct with only %d of %d required shards failed as expected: %v\n",
		len(short), k, err)

	fmt.Printf("demo finished in %s\n", time.Since(start).Round(time.Millisecond))
}
