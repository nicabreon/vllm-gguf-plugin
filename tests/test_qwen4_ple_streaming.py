#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone CPU-only numeric/streaming checks; imports no vLLM package.

Run with the plugin's numpy/torch/gguf environment:
  HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
    VLLM_PLUGINS='' python tests/test_qwen4_ple_streaming.py

All GGUF fixtures are tiny and temporary. No checkpoint download or GPU call.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / "vllm_gguf_plugin" / "qwen4_ple.py"
SPEC = importlib.util.spec_from_file_location("qwen4_ple_standalone", SOURCE)
assert SPEC is not None and SPEC.loader is not None
PLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLE
SPEC.loader.exec_module(PLE)

SMALL_CONFIG = {
    "ple_layer_ids": [2],
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ple_embed_dim": 640,
    "ngram_vocab_size_base": 17,
    "make_ngram_vocab_size_divisible_by": 8,
    "split_ngram_parts": 3,
    "vocab_size": 248320,
    "seed": 1234,
    "ple_embedding_dtype": None,
}
PREFIX = "model.layers.1.ple.ple_embedding"
GOLDEN_MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
GOLDEN_METADATA = {
    "qwen4exp.ple.layers": [1],
    "qwen4exp.ple.ngram_size": 3,
    "qwen4exp.ple.heads_per_ngram": 2,
    "qwen4exp.ple.layer_multipliers": GOLDEN_MULTIPLIERS,
    "qwen4exp.ple.head_offsets": [0, 17, 36, 59],
    "qwen4exp.ple.head_vocab_sizes": [17, 19, 23, 29],
}


def assert_rejected(call, contains):
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        assert contains in str(error), (contains, str(error))
    else:
        raise AssertionError("expected rejection: " + contains)


def packed_fixture(rows=88):
    blocks = rows * 5  # 160 values per row, 32 values per IQ4_NL block.
    raw = np.empty((blocks, 18), dtype=np.uint8)
    scales = np.array(
        [0.0, -0.0, 1.0, -0.25, 2.0**-24, 2.0**-14, 16.0, 65504.0], dtype="<f2"
    )
    raw[:, :2] = np.resize(scales, blocks).view(np.uint8).reshape(blocks, 2)
    raw[:, 2:] = (
        (np.arange(blocks * 16, dtype=np.uint64) * 37 % 256)
        .astype(np.uint8)
        .reshape(blocks, 16)
    )
    return raw.reshape(rows, 90)


def write_gguf(path, *, metadata=None, packed=None, quant_type=None):
    writer = gguf.GGUFWriter(path, arch="qwen4exp")
    for key, value in (metadata or {}).items():
        if isinstance(value, list):
            # GGUF0.19 infers Python-int arrays as INT32, while native PLE hash
            # multipliers deliberately exceed 2**31. Preserve their exact bits.
            writer.add_key_value(
                key,
                value,
                gguf.GGUFValueType.ARRAY,
                sub_type=gguf.GGUFValueType.UINT64,
            )
        else:
            writer.add_uint32(key, value)
    if packed is not None:
        writer.add_tensor(
            PLE.PLE_TENSOR_NAME,
            packed,
            raw_dtype=quant_type or gguf.GGMLQuantizationType.IQ4_NL,
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    if packed is not None:
        writer.write_tensors_to_file()
    writer.close()


def main():
    checks = []
    layout = PLE.qwen4_ple_layout(SMALL_CONFIG)
    assert (
        layout.prefix,
        layout.embedding_dim,
        layout.total_rows,
        layout.shard_rows,
    ) == (PREFIX, 160, 88, 30)
    assert list(layout.layer_multipliers) == GOLDEN_MULTIPLIERS
    assert list(layout.head_vocab_sizes) == [17, 19, 23, 29]
    checks.append(
        "native geometry/hash buffers match fixed independent small-config goldens"
    )

    config_wrapper = SimpleNamespace(hf_text_config=SimpleNamespace(**SMALL_CONFIG))
    assert PLE.qwen4_ple_layout(config_wrapper) == layout
    hf_wrapper = SimpleNamespace(
        hf_config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(**SMALL_CONFIG)
        )
    )
    assert PLE.qwen4_ple_layout(hf_wrapper) == layout
    checks.append("ModelConfig.hf_text_config and HF get_text_config inputs supported")

    production = dict(
        SMALL_CONFIG,
        heads_per_ngram=8,
        ple_embed_dim=2560,
        ngram_vocab_size_base=20000000,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=128,
    )
    production_layout = PLE.qwen4_ple_layout(production)
    assert production_layout.total_rows == 320001536
    assert production_layout.shard_rows == 2500012
    assert production_layout.embedding_dim == 160
    assert production_layout.shard_bytes == 800003840
    assert production_layout.total_rows * 160 * 2 == 102400491520
    checks.append(
        "exact production128-shard geometry checked without allocating its table"
    )

    packed = packed_fixture()
    reference = gguf.dequantize(packed, gguf.GGMLQuantizationType.IQ4_NL)
    decoded = PLE._decode_iq4_nl_rows(packed, 160)
    np.testing.assert_array_equal(decoded.view(np.uint32), reference.view(np.uint32))
    # Explicit row160 and awkward row boundaries, independently dequantized.
    for start, end in [(0, 1), (1, 2), (7, 9), (29, 31), (59, 61), (87, 88)]:
        part = PLE._decode_iq4_nl_rows(packed[start:end], 160)
        np.testing.assert_array_equal(
            part.view(np.uint32), reference[start:end].view(np.uint32)
        )
    checks.append(
        "IQ4_NL F32 decode matches gguf reference bitwise, "
        "including row160 and boundary slices"
    )

    with tempfile.TemporaryDirectory(prefix="qwen4-ple-cpu-") as temporary:
        root = Path(temporary)
        metadata_file, weights_file = root / "metadata.gguf", root / "weights.gguf"
        write_gguf(metadata_file, metadata=GOLDEN_METADATA)
        write_gguf(weights_file, packed=packed)
        files = [metadata_file, weights_file]
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        expected_bf16 = torch.from_numpy(reference).to(torch.bfloat16)
        for row_chunk in (1, 7, 16, 31, PLE.DEFAULT_ROW_CHUNK):
            outputs, pointers, names = [], [], []
            buffers = {}
            for name, value in PLE.iter_qwen4_ple_weights(
                files, config_wrapper, row_chunk=row_chunk
            ):
                assert value.device.type == "cpu"
                if value.dtype == torch.int64:
                    buffers[name] = value.tolist()
                    continue
                assert value.dtype == torch.bfloat16
                pointers.append(value.data_ptr())
                names.append(name)
                outputs.append(
                    value.clone()
                )  # The documented native-loader copy contract.
            assert buffers == {
                PREFIX + ".layer_multipliers": GOLDEN_MULTIPLIERS,
                PREFIX + ".ngram_heads_offsets": [0, 17, 36, 59],
                PREFIX + ".ngram_heads_vocab_sizes": [17, 19, 23, 29],
            }
            assert names == [
                PREFIX + f".ngram_embedding.shard_{index}.weight" for index in range(3)
            ]
            assert [tuple(tensor.shape) for tensor in outputs] == [
                (30, 160),
                (30, 160),
                (28, 160),
            ]
            assert len(set(pointers)) == 1, (
                "must reuse exactly one BF16 shard allocation"
            )
            assert torch.equal(
                torch.cat(outputs).view(torch.int16), expected_bf16.view(torch.int16)
            )
        assert hashes == {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        }
        checks.append(
            "real metadata-only+packed GGUF files stream exact BF16 native shards "
            "at five chunk sizes"
        )
        checks.append(
            "one reusable shard allocation; native names, tail rows "
            "and INT64 buffers correct"
        )
        checks.append("source GGUF bytes remain unchanged")

        for invalid in (0, -1, True):
            assert_rejected(
                lambda invalid=invalid: next(
                    PLE.iter_qwen4_ple_weights(files, SMALL_CONFIG, row_chunk=invalid)
                ),
                "row_chunk",
            )
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights(files, SMALL_CONFIG, row_chunk=1000000)
            ),
            "FP32",
        )
        assert_rejected(
            lambda: next(PLE.iter_qwen4_ple_weights(str(weights_file), SMALL_CONFIG)),
            "nonempty sequence",
        )
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights([weights_file, weights_file], SMALL_CONFIG)
            ),
            "duplicate GGUF",
        )
        assert_rejected(
            lambda: next(PLE.iter_qwen4_ple_weights([weights_file], SMALL_CONFIG)),
            "metadata is missing",
        )
        checks.append(
            "invalid chunk budgets, duplicate/missing paths and missing metadata "
            "rejected before allocation"
        )

        bad_metadata = copy.deepcopy(GOLDEN_METADATA)
        bad_metadata["qwen4exp.ple.head_offsets"][1] += 1
        bad_meta_file = root / "bad-metadata.gguf"
        write_gguf(bad_meta_file, metadata=bad_metadata)
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights([bad_meta_file, weights_file], SMALL_CONFIG)
            ),
            "head_offsets",
        )
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights(
                    [metadata_file, bad_meta_file, weights_file], SMALL_CONFIG
                )
            ),
            "disagree",
        )
        duplicate_file = root / "duplicate.gguf"
        shutil.copyfile(weights_file, duplicate_file)
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights(
                    [metadata_file, weights_file, duplicate_file], SMALL_CONFIG
                )
            ),
            "exactly one",
        )
        wrong_shape = root / "wrong-shape.gguf"
        write_gguf(wrong_shape, packed=packed_fixture(87))
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights([metadata_file, wrong_shape], SMALL_CONFIG)
            ),
            "shape mismatch",
        )
        wrong_type = root / "wrong-type.gguf"
        write_gguf(wrong_type, packed=packed, quant_type=gguf.GGMLQuantizationType.Q4_0)
        assert_rejected(
            lambda: next(
                PLE.iter_qwen4_ple_weights([metadata_file, wrong_type], SMALL_CONFIG)
            ),
            "must use IQ4_NL",
        )
        checks.append(
            "wrong/conflicting metadata, duplicate PLE tensors "
            "and shape/type mismatches rejected"
        )

    assert_rejected(
        lambda: PLE.qwen4_ple_layout(dict(SMALL_CONFIG, ple_layer_ids=[1, 2])),
        "exactly one",
    )
    assert_rejected(
        lambda: PLE.qwen4_ple_layout(
            dict(SMALL_CONFIG, ple_embedding_dtype="float8_e4m3fn")
        ),
        "requires ple_embedding_dtype=None",
    )
    checks.append(
        "ambiguous multi-PLE or non-BF16 native storage configuration rejected"
    )
    print(
        json.dumps(
            {
                "ok": True,
                "checks": checks,
                "count": len(checks),
                "gpu_execution": False,
                "checkpoint_downloads": False,
                "production_table_allocated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
