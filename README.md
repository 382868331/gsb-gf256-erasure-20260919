# 系统式分片纠删码库 (GF(256) Erasure Coding)

离线文件分片后在**已知哪些分片丢失**（已知擦除）的情况下恢复原始对象。仅标准库，无第三方依赖，Windows 原生离线运行（Go 1.26.5）。

## 原理

- 域：GF(256)，域多项式 `0x11d`，加法为 XOR，生成元 `0x02`。
- 构造 n×k Vandermonde 矩阵 `V`：0 起始行 `i` 取域元素 `i+1`，`V[i][j] = (i+1)^j`。
- 生成矩阵 `G = V · inv(V 的前 k 行)`，因此 `G` 的前 k 行是单位阵（系统式：前 k 片即原始数据分片）。
- 编码：每个校验片 = `G` 对应行与 k 个数据片的域线性组合。
- 恢复：任取 k 个可用片，取 `G` 对应 k 行组成方阵求逆（Gauss-Jordan），解出 k 个数据片，再重算全部 n 片。

参数约束：`1 <= k <= 16`，`1 <= m <= 8`，`n = k+m`，输入至多 4 MiB。原长 `N` 决定片长 `s = ceil(N/k)`，数据连续分为 k 片并在末尾补零；空输入 `s = 0`（所有片为零长，零长 bytes 是有效分片）。

## 接口

```go
import erasure "github.com/382868331/gsb-gf256-erasure-20260919"

// 编码：返回 n = k+m 个分片（下标即分片索引），输出与输入互不别名。
shards, err := erasure.Encode(data []byte, k, m int) ([][]byte, error)

// 恢复：从任意至少 k 个可用分片重建全部 n 片及原数据。
// 缺失索引不出现在列表中；索引从 0 开始；origLen 为原长 N。
type Shard struct { Index int; Data []byte }
all, data, err := erasure.Reconstruct(shards []Shard, k, m, origLen int) ([][]byte, []byte, error)
```

`Reconstruct` 的校验与拒绝规则：

- 拒绝：重复/越界索引、负原长、原长超过 4 MiB、片长不等于 `ceil(N/k)`、少于 k 片、`k`/`m` 越界。
- 解码后检查恢复数据尾部补零必须为零，并用重建结果复核所有已给分片，不一致则报错。
- 出错时不修改输入，返回 nil；输出字节与输入互不别名。
- 注意：恰好 k 片时的不一致性不可普遍检测，本库不对此做虚假保证。

错误为可 `errors.Is` 匹配的哨兵错误：`ErrInvalidK`、`ErrInvalidM`、`ErrInputTooLarge`、`ErrNegativeLength`、`ErrTooFewShards`、`ErrDuplicateIndex`、`ErrIndexOutOfRange`、`ErrBadShardLen`、`ErrBadPadding`、`ErrInconsistent`。

## 运行

演示（约 8 秒内完成，展示双片丢失恢复与不足片失败）：

```
go run ./cmd/demo
```

测试：

```
go test ./... -count=1 -timeout=60s
```

测试覆盖：域乘法对独立逐位参考的全量核对、生成矩阵系统性、空输入/零长分片、固定种子非整除长度往返、k<=3 且 m<=3 时所有大小为 k 的可用索引组合穷举、数据/校验片缺失、坏原长/片长、重复/越界索引、额外片不一致、非零补零检测、输入不变与输出不别名、4 MiB 边界。

## 文件

- `gf.go` — GF(256) 域运算（exp/log 表 + 全量乘法表）
- `matrix.go` — Vandermonde 构造、矩阵乘法、Gauss-Jordan 求逆、生成矩阵
- `erasure.go` — `Encode` / `Reconstruct` 及参数与一致性校验
- `erasure_test.go` — 测试
- `cmd/demo/main.go` — 演示
