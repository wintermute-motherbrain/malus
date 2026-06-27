# M22 — Data I/O + nanoGPT Capstone

**Crates:** `malus-runtime`, `malus-sema`, `malus-codegen-cpu`, `malus-cli`.

Add a minimal file-read surface so malus programs can load real text data, then implement `examples/nanogpt.ml` — a decoder-only transformer that reads tiny Shakespeare, trains with AdamW and autograd, shows decreasing loss, and samples text. This is the V3 capstone. See ADR-0018.

## Done-When

`examples/nanogpt.ml` runs on an M-series Mac with `data/tiny_shakespeare.txt` present:

1. Reads the file, builds a character vocab, encodes the full text as an `Array` of `Tensor<i32>` batches.
2. Trains a decoder-only transformer (≥2 blocks, ≥32-dim, causal self-attention + MLP + LayerNorm, AdamW) showing **decreasing cross-entropy loss** over the training run (printed every N steps).
3. After training, samples and prints a plausible character-level text continuation of a fixed prompt.

The run completes in a reasonable wall-clock time on an M-series Mac (target: ≤10 minutes for a visible result). Model scale is tuned accordingly (small config with real architecture).

## Scope

### 1. File Read Builtin

**Builtins (`malus-sema/src/builtins.rs`):** Register `read_file(path: str) -> str` — reads a UTF-8 text file from disk and returns its contents as a string value. This is the only file I/O primitive in V3; see ADR-0018.

**Runtime (`malus-runtime/src/lib.rs`):** `extern "C" fn malus_read_file(path_ptr: *const u8, path_len: usize, out_len: *mut usize) -> *const u8` — calls `std::fs::read_to_string`, leaks the string into a heap buffer, writes its length to `out_len`, and returns the pointer. The caller is responsible for treating this as a borrowed view (the buffer is not freed in V3 — whole-program lifetime).

**Codegen-cpu (`malus-codegen-cpu/src/lib.rs`):** Lower `read_file(path)` to a JIT call to `malus_read_file`. The return type in sema is `ResolvedTy::Str`; codegen represents it as a `(ptr: i64, len: i64)` pair in a two-slot Cranelift SSA value.

**String operations needed for char tokenization:** Register `str_len(s: str) -> i32`, `str_char_at(s: str, i: i32) -> i32` (returns Unicode codepoint), and `str_from_char(c: i32) -> str`. These are the minimum primitives for building a vocab map and encoding the dataset.

### 2. In-Language Char Tokenization

The nanogpt example builds its vocabulary and tokenizer entirely in malus using `str_len`, `str_char_at`, and existing control-flow and array features. No tokenizer library. The vocab is a `FixedArray` of characters; encoding walks the text and looks up indices. Batch construction uses `ForIn` and indexed assignment (`a[i] = e` from M20).

### 3. `examples/nanogpt.ml`

Write the capstone example using all features from V2 and V3:

**Architecture (decoder-only transformer):**

```malus
struct Config:
    vocab_size: i32
    ctx_len:    i32
    n_embd:     i32
    n_head:     i32
    n_layer:    i32

struct Block:
    ln1_scale: Variable<f32>
    ln1_bias:  Variable<f32>
    wq: Variable<f32>
    wk: Variable<f32>
    wv: Variable<f32>
    wo: Variable<f32>
    ln2_scale: Variable<f32>
    ln2_bias:  Variable<f32>
    mlp_w1: Variable<f32>
    mlp_w2: Variable<f32>

struct GPT:
    tok_emb: Variable<f32>
    pos_emb: Variable<f32>
    blocks:  Array<Block, 2>
    ln_f_scale: Variable<f32>
    ln_f_bias:  Variable<f32>
    lm_head: Variable<f32>
```

The `forward` fn takes a `GPT`, token indices (`Tensor<i32>`), and returns `Variable<f32>` logits. A training loop runs with AdamW (from M20's stdlib), calling `backward(loss)` and `adamw_step`. Sampling uses `softmax` + a greedy or temperature-scaled argmax.

**Data flow:**

```
read_file("data/tiny_shakespeare.txt")
  → char_tokenize (build vocab, encode as i32 array)
  → batch_iter (yield (B, T) input and target integer tensors each step)
  → forward → cross_entropy → backward → adamw_step
```

## Out of Scope

- SafeTensors / model checkpoint save and load (post-V3)
- BPE or wordpiece tokenization (post-V3)
- Streaming file reads or large-file handling (post-V3)
- Multi-file or directory datasets (post-V3)
- Sampling with nucleus/top-k filtering (post-V3)
- Flash attention (post-V3)
- Multi-GPU (post-V3)
