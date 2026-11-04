# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU conversion of Qwen4Exp's packed IQ4_NL PLE checkpoint table.

The output names target the native Qwen4ExpForCausalLM implementation pinned at
vLLM 5689c2790c911a55c75825aac5166305df0b591d. No model-sized dense copy or
intermediate checkpoint is created. The BF16 shard storage is reused between
yields: a consumer MUST copy each shard synchronously before advancing the
iterator, as the native Qwen4 PLE weight loader does. Do not list()/cache it.

This module deliberately imports no vLLM modules and performs no GPU work.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

PLE_TENSOR_NAME = "per_layer_token_embd.weight"
DEFAULT_ROW_CHUNK = 65536
MAX_SHARD_BYTES = 1 << 30
MAX_FP32_CHUNK_BYTES = 128 << 20
_IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _text_config(model_config: Any) -> Any:
    text = _value(model_config, "hf_text_config")
    if text is not None:
        return text
    config = _value(model_config, "hf_config", model_config)
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return _value(config, "text_config", config)


def _positive_int(value: Any, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _is_prime_64(value: int) -> bool:
    # Same deterministic 64-bit Miller-Rabin bases as the pinned native PLE.
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent, shifts = value - 1, 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _head_vocab_sizes(base: int, count: int) -> tuple[int, ...]:
    sizes = []
    candidate = base
    while len(sizes) < count:
        if _is_prime_64(candidate):
            sizes.append(candidate)
        candidate += 1
    return tuple(sizes)


def _splitmix64(value: int) -> int:
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


@dataclass(frozen=True)
class Qwen4PLELayout:
    """Native table geometry and exact hash buffers for one PLE layer."""

    prefix: str
    embedding_dim: int
    total_rows: int
    num_shards: int
    shard_rows: int
    ngram_size: int
    heads_per_ngram: int
    layer_id: int
    layer_multipliers: tuple[int, ...]
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]

    @property
    def shard_bytes(self) -> int:
        return self.shard_rows * self.embedding_dim * 2


def qwen4_ple_layout(model_config: Any) -> Qwen4PLELayout:
    """Resolve ModelConfig, HF config/get_text_config(), or a plain text config.

    Only the single-PLE-layer layout used by this checkpoint is accepted. The
    packed GGUF contains one table and cannot disambiguate several PLE tables.
    """
    config = _text_config(model_config)
    layer_ids = _value(config, "ple_layer_ids")
    if not isinstance(layer_ids, (list, tuple)) or len(layer_ids) != 1:
        raise ValueError(
            "packed Qwen4 PLE currently requires exactly one ple_layer_ids entry"
        )
    layer_id = _positive_int(layer_ids[0], "ple_layer_ids[0]") - 1
    ngram_size = _positive_int(_value(config, "ngram_size"), "ngram_size")
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    heads_per_ngram = _positive_int(
        _value(config, "heads_per_ngram"), "heads_per_ngram"
    )
    heads = (ngram_size - 1) * heads_per_ngram
    ple_dim = _positive_int(_value(config, "ple_embed_dim"), "ple_embed_dim")
    if ple_dim % heads:
        raise ValueError("ple_embed_dim must be divisible by the number of ngram heads")
    embedding_dim = ple_dim // heads
    if embedding_dim % 32:
        raise ValueError(
            "IQ4_NL PLE rows must contain a whole number of 32-value blocks"
        )
    if _value(config, "ple_embedding_dtype") is not None:
        raise ValueError(
            "BF16 PLE conversion requires ple_embedding_dtype=None in native config"
        )
    divisor = _positive_int(
        _value(config, "make_ngram_vocab_size_divisible_by"),
        "make_ngram_vocab_size_divisible_by",
    )
    num_shards = _positive_int(
        _value(config, "split_ngram_parts", 512), "split_ngram_parts"
    )
    base = _positive_int(
        _value(config, "ngram_vocab_size_base"), "ngram_vocab_size_base"
    )
    sizes = _head_vocab_sizes(base, heads)
    offsets, rows = [], 0
    for size in sizes:
        offsets.append(rows)
        rows += size
    total_rows = (rows + divisor - 1) // divisor * divisor
    shard_rows = (total_rows + num_shards - 1) // num_shards
    vocab_size = _positive_int(_value(config, "vocab_size"), "vocab_size")
    half_bound = max(1, (((1 << 63) - 1) // vocab_size) // 2)
    seed = _value(config, "seed", 1234)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    # With one PLE layer the native ple_dense_layer_id is zero, independent of
    # its transformer layer index; its layer-prime contribution is therefore 0.
    multipliers = tuple(
        2 * (_splitmix64(int(seed) + _GAMMA * (index + 1)) % half_bound) + 1
        for index in range(ngram_size)
    )
    layout = Qwen4PLELayout(
        prefix=f"model.layers.{layer_id}.ple.ple_embedding",
        embedding_dim=embedding_dim,
        total_rows=total_rows,
        num_shards=num_shards,
        shard_rows=shard_rows,
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        layer_id=layer_id,
        layer_multipliers=multipliers,
        head_offsets=tuple(offsets),
        head_vocab_sizes=sizes,
    )
    if layout.shard_bytes > MAX_SHARD_BYTES:
        raise ValueError(
            f"PLE shard would allocate {layout.shard_bytes} bytes, exceeding the "
            f"bounded CPU limit {MAX_SHARD_BYTES}"
        )
    return layout


def qwen4_ple_prefix(model_config: Any) -> str:
    """Return the native PLE module prefix, before `.ngram_embedding`."""
    return qwen4_ple_layout(model_config).prefix


def _field_value(readers: Sequence[Any], name: str) -> Any:
    values = [
        reader.fields[name].contents() for reader in readers if name in reader.fields
    ]
    if not values:
        raise ValueError(f"required GGUF metadata is missing: {name}")
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"GGUF shards disagree on metadata: {name}")
    return values[0]


def _validate_metadata(readers: Sequence[Any], layout: Qwen4PLELayout) -> None:
    if _field_value(readers, "general.architecture") != "qwen4exp":
        raise ValueError("PLE conversion requires GGUF architecture qwen4exp")
    expected = {
        "qwen4exp.ple.layers": [layout.layer_id],
        "qwen4exp.ple.ngram_size": layout.ngram_size,
        "qwen4exp.ple.heads_per_ngram": layout.heads_per_ngram,
        "qwen4exp.ple.layer_multipliers": list(layout.layer_multipliers),
        "qwen4exp.ple.head_offsets": list(layout.head_offsets),
        "qwen4exp.ple.head_vocab_sizes": list(layout.head_vocab_sizes),
    }
    for key, value in expected.items():
        if _field_value(readers, key) != value:
            raise ValueError(f"GGUF {key} disagrees with the native text configuration")


def _decode_iq4_nl_rows(packed: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Decode whole IQ4_NL rows to F32, preserving GGUF nibble order exactly."""
    if packed.dtype != np.uint8 or packed.ndim != 2:
        raise ValueError("packed IQ4_NL rows must be a 2D uint8 array")
    row_bytes = embedding_dim // 32 * 18
    if embedding_dim % 32 or packed.shape[1] != row_bytes:
        raise ValueError("packed IQ4_NL row width is inconsistent with embedding_dim")
    blocks = packed.reshape(-1, 18)
    # IQ4_NL is a little-endian FP16 scale followed by 16 packed nibbles.
    scales = blocks[:, :2].copy().view("<f2").reshape(-1, 1).astype(np.float32)
    quants = blocks[:, 2:]
    decoded = np.empty((blocks.shape[0], 32), dtype=np.float32)
    np.multiply(_IQ4_NL_VALUES[quants & 15], scales, out=decoded[:, :16])
    np.multiply(_IQ4_NL_VALUES[quants >> 4], scales, out=decoded[:, 16:])
    return decoded.reshape(packed.shape[0], embedding_dim)


def iter_qwen4_ple_weights(
    gguf_filepaths: Sequence[str | PathLike[str]],
    model_config: Any,
    *,
    row_chunk: int = DEFAULT_ROW_CHUNK,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield native INT64 PLE buffers and reusable BF16 CPU checkpoint shards.

    Requires all local GGUF backbone paths, including a metadata-only first
    shard. Reader arrays are read-only memory maps. There is one reusable BF16
    shard buffer (Huihui: 800,003,840 bytes) plus bounded F32 decode chunks.
    File integrity/authenticity remains the caller's responsibility.
    """
    layout = qwen4_ple_layout(model_config)
    row_chunk = _positive_int(row_chunk, "row_chunk")
    if row_chunk * layout.embedding_dim * 4 > MAX_FP32_CHUNK_BYTES:
        raise ValueError("row_chunk exceeds the bounded FP32 decode-chunk limit")
    if isinstance(gguf_filepaths, (str, bytes, PathLike)) or not gguf_filepaths:
        raise ValueError("gguf_filepaths must be a nonempty sequence of backbone paths")
    paths = [Path(path).expanduser().resolve(strict=True) for path in gguf_filepaths]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate GGUF file paths")
    readers = [gguf.GGUFReader(path, mode="r") for path in paths]
    _validate_metadata(readers, layout)
    matches = [
        tensor
        for reader in readers
        for tensor in reader.tensors
        if tensor.name == PLE_TENSOR_NAME
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one packed PLE tensor, found {len(matches)}"
        )
    tensor = matches[0]
    expected_shape = (layout.embedding_dim, layout.total_rows)
    if tuple(int(value) for value in tensor.shape) != expected_shape:
        raise ValueError(
            f"PLE GGML shape mismatch: expected {expected_shape}, "
            f"got {tuple(int(value) for value in tensor.shape)}"
        )
    if int(tensor.tensor_type) != int(gguf.GGMLQuantizationType.IQ4_NL):
        raise ValueError("packed PLE tensor must use IQ4_NL")
    row_bytes = layout.embedding_dim // 32 * 18
    expected_bytes = layout.total_rows * row_bytes
    if int(tensor.n_bytes) != expected_bytes:
        raise ValueError("packed PLE byte count does not match the validated shape")
    packed = np.asarray(tensor.data)
    if (
        packed.dtype != np.uint8
        or not packed.flags.c_contiguous
        or packed.size != expected_bytes
    ):
        raise ValueError("packed PLE data must be a contiguous uint8 memory-map view")
    packed = packed.reshape(layout.total_rows, row_bytes)

    for suffix, values in (
        ("layer_multipliers", layout.layer_multipliers),
        ("ngram_heads_offsets", layout.head_offsets),
        ("ngram_heads_vocab_sizes", layout.head_vocab_sizes),
    ):
        yield (
            f"{layout.prefix}.{suffix}",
            torch.tensor(values, dtype=torch.int64, device="cpu"),
        )

    # Allocate once: retaining the previous yielded tuple while requesting the
    # next item must not cause two simultaneous model-sized shard allocations.
    storage = torch.empty(
        (layout.shard_rows, layout.embedding_dim), dtype=torch.bfloat16, device="cpu"
    )
    for shard_index in range(layout.num_shards):
        start = shard_index * layout.shard_rows
        rows = max(0, min(layout.shard_rows, layout.total_rows - start))
        shard = storage[:rows]
        for local_start in range(0, rows, row_chunk):
            count = min(row_chunk, rows - local_start)
            decoded = _decode_iq4_nl_rows(
                packed[start + local_start : start + local_start + count],
                layout.embedding_dim,
            )
            shard[local_start : local_start + count].copy_(torch.from_numpy(decoded))
            del decoded
        yield f"{layout.prefix}.ngram_embedding.shard_{shard_index}.weight", shard
