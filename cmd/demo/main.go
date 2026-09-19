// Command demo shows a successful recovery from two lost shards and a
// real failure caused by providing fewer than k shards. All numbers are
// computed by the library at run time.
package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"math/rand"
	"os"

	"github.com/382868331/gsb-gf256-erasure-20260919/erasure"
)

func main() {
	failed := false

	// --- Scenario 1: lose two shards (one data, one parity), recover. ---
	k, m := 8, 4
	n := k + m
	rng := rand.New(rand.NewSource(20260919))
	data := make([]byte, 100003) // non-divisible by k on purpose
	rng.Read(data)

	shards, err := erasure.Encode(data, k, m)
	if err != nil {
		fmt.Println("encode failed:", err)
		os.Exit(1)
	}
	fmt.Printf("[1] encoded %d bytes into n=%d shards (k=%d, m=%d), shard size %d\n",
		len(data), n, k, m, len(shards[0]))

	// Shard 1 (data) and shard 9 (parity) are lost; the rest survive.
	lost := map[int]bool{1: true, 9: true}
	var avail []erasure.Shard
	for i, sh := range shards {
		if lost[i] {
			continue
		}
		avail = append(avail, erasure.Shard{Index: i, Data: sh})
	}
	fmt.Printf("[1] lost shards 1 (data) and 9 (parity); %d shards remain\n", len(avail))

	all, recovered, err := erasure.Reconstruct(len(data), avail, k, m)
	if err != nil {
		fmt.Println("[1] FAIL: reconstruct returned error:", err)
		failed = true
	} else {
		okData := bytes.Equal(recovered, data)
		okShards := true
		for i := 0; i < n; i++ {
			if !bytes.Equal(all[i], shards[i]) {
				okShards = false
			}
		}
		fmt.Printf("[1] recovered %d bytes, sha256=%x\n", len(recovered), sha256.Sum256(recovered))
		fmt.Printf("[1] original sha256=%x\n", sha256.Sum256(data))
		if okData && okShards {
			fmt.Println("[1] OK: recovered data and all 12 rebuilt shards match the originals")
		} else {
			fmt.Println("[1] FAIL: recovered bytes differ from the original")
			failed = true
		}
	}

	// --- Scenario 2: fewer than k shards, recovery must fail. ---
	tooFew := avail[:k-1] // only k-1 = 7 shards
	fmt.Printf("\n[2] attempting recovery from only %d shards (need at least k=%d)\n", len(tooFew), k)
	_, _, err = erasure.Reconstruct(len(data), tooFew, k, m)
	if err != nil {
		fmt.Printf("[2] OK: recovery correctly rejected: %v\n", err)
	} else {
		fmt.Println("[2] FAIL: recovery unexpectedly succeeded with insufficient shards")
		failed = true
	}

	if failed {
		os.Exit(1)
	}
	fmt.Println("\ndemo finished: one successful recovery, one correctly triggered failure")
}
