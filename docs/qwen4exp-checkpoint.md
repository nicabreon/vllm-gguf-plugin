# Qwen4Exp text GGUF development checkpoint

This branch preserves the initial text-only adapter and compatibility fixes for
the mixed-quantized Huihui Qwen3.8 Flash Next abliterated v2 UD-IQ4_XS checkpoint.
It starts from plugin commit `e454affe0ca23c9dc8754ec3033270cb30622ed5` and targets
the native Qwen4Exp implementation in vLLM commit
`5689c2790c911a55c75825aac5166305df0b591d`.

## Scope

- Map the checkpoint's GDN, hyperconnection, sparse-attention indexer, experts
  and PLE tensors to native text-model names.
- Keep ordinary quantized matrices and expert weights packed.
- Stream the IQ4_NL PLE table through bounded CPU buffers into BF16 checkpoint
  shards and reconstruct the native PLE hash buffers from checked metadata.
- Handle newer Transformers configuration imports and module names containing
  the word `weight` when emitting quantization metadata.
- Guard IQ4_NL dequantization launches for complete 32-value quant blocks that
  do not fill a 256-value launch block, including rows of width 160.
- Allow explicitly skipping the native extension for isolated fallback builds.

The adapter is intentionally restricted to text-only checkpoints with one PLE
layer and the supported metadata/layout. The registered Qwen4 model types do
not imply support for every checkpoint, vision input or PLE quantization.
The hyperconnection and indexer mappings follow the pinned native model.

## PLE loader contract

The PLE generator reuses each BF16 shard buffer. Its consumer **must copy the
shard synchronously before requesting the next item**. The pinned native vLLM
loader follows this contract. Buffering the iterator, calling `list()` on it,
or reusing the adapter with an asynchronous consumer can corrupt the table.
Compatibility with another vLLM revision must be checked before serving.

The full table is kept in host RAM; the tested table expands to
102,400,491,520 bytes in BF16. Streaming conversion bounds temporary buffers,
but it does not remove that native runtime memory requirement.

## Validation and performance status

The deployed checkpoint has passed synthetic arithmetic, automatic tool call
and tool round-trip checks on AMD MI300X and NVIDIA RTX PRO 6000 Blackwell.
These short canaries do not establish full agentic or long-context quality.

The initial MI300X decode-graph run measured about 79.69 decode tokens/s for
one request and 422.78 decode tokens/s aggregate for 64 concurrent requests,
using short raw-completion prompts (32 input and 512 output tokens). Main
attention KV was FP8. These are development measurements, not hardware peaks
or a production capacity guarantee. **The 1,000 aggregate decode TPS target
has not been reached by this GGUF checkpoint.** Earlier 1,000+ TPS results used
a different safetensors model and kernel path.

The rows-per-block kernel variants and further MoE optimizations remain
separate experiments; they are not enabled or included in this checkpoint.

## Reproducing CPU checks

With the plugin's compatible `torch`, `numpy`, `gguf` and vLLM environment:

```bash
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  VLLM_PLUGINS='' python tests/test_qwen4_ple_streaming.py

CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  python tests/test_qwen4_exp_adapter_cpu.py \
  --inventory /path/to/tensor-inventory.json --config /path/to/model-config.json
```

The streaming tests generate small temporary GGUF fixtures. The adapter tests
include independent conversion fixtures; the exact full-checkpoint inventory
check is skipped unless its external inventory and config are supplied.
No model weights, credentials, deployment logs or private configuration are
included in this branch. Further work before an upstream PR includes portable
CI coverage, native-kernel regression tests and review against current upstream.
