# SPDX-License-Identifier: Apache-2.0
"""CPU-only adapter checks; run as a standalone script without pytest.

python tests/test_qwen4_exp_adapter_cpu.py \
  --inventory /path/to/huihui-tensors.json --config /path/to/model-config.json

No model weights or GPU allocations are used.  Inventory assertions cover the
actual checkpoint; numerical fixtures independently encode the converter's math.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gguf
import numpy as np
import torch

from vllm_gguf_plugin.gguf_files import GGUFModelFiles
from vllm_gguf_plugin.weights_adapter.qwen4_exp import (
    PLE_GGUF_NAME,
    Qwen4ExpGGUFAdapter,
    expected_qwen4_tensor_names,
    qwen4_ple_prefix,
)


_ARGS = None


def config_for(text):
    return SimpleNamespace(hf_config=SimpleNamespace(get_text_config=lambda: text))


def small_config():
    return config_for(SimpleNamespace(
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=2, linear_value_head_dim=2,
        ple_layer_ids=[2],
    ))


def ggml_tile(values, dim, *, key_heads=2, repeats=3, head_dim=2):
    """Independent forward conversion: grouped by K -> repeated K tiles."""
    shape = list(values.shape)
    grouped = shape[:dim] + [key_heads, repeats, head_dim] + shape[dim + 1:]
    return values.reshape(grouped).transpose(dim, dim + 1).reshape(shape).contiguous()


class Qwen4AdapterCPUChecks(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen4ExpGGUFAdapter()
        self.config = small_config()

    def run_weights(self, weights):
        return dict(self.adapter.transform_weights(weights, self.config))

    def test_exact_checkpoint_mapping(self):
        if _ARGS is None or not _ARGS.inventory or not _ARGS.config:
            self.skipTest("Supply --inventory and --config to check the actual checkpoint")
        manifest = json.loads(Path(_ARGS.inventory).read_text())
        tensors = [tensor for file in manifest["files"] for tensor in file["tensors"]]
        raw = json.loads(Path(_ARGS.config).read_text())
        text = SimpleNamespace(**raw.get("text_config", raw))
        names = {tensor["name"] for tensor in tensors}
        self.assertEqual(len(names), 1224)
        self.assertEqual(expected_qwen4_tensor_names(text), names)
        with patch(
            "vllm_gguf_plugin.weights_adapter.qwen4_exp.get_gguf_tensor_names",
            return_value=names,
        ):
            mapping = self.adapter.build_name_map(GGUFModelFiles(("fixture.gguf",)), config_for(text))
        self.assertEqual(len(mapping), 1223)
        self.assertNotIn(PLE_GGUF_NAME, mapping)
        self.assertEqual(len(set(mapping.values())), 1223)
        self.assertEqual(mapping["blk.3.ffn_down_exps.weight"], "model.layers.3.mlp.experts.0.down_proj.weight")
        self.assertEqual(mapping["blk.1.ple_key.weight"], "model.layers.1.ple.key_proj.weight")
        self.assertEqual(qwen4_ple_prefix(config_for(text)), "model.layers.1.ple.ple_embedding")
        with patch(
            "vllm_gguf_plugin.weights_adapter.qwen4_exp.get_gguf_tensor_names",
            return_value=names - {"blk.3.attn_q.weight"},
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing Qwen4Exp"):
                self.adapter.build_name_map(GGUFModelFiles(("fixture.gguf",)), config_for(text))
        with patch(
            "vllm_gguf_plugin.weights_adapter.qwen4_exp.get_gguf_tensor_names",
            return_value=names | {"blk.3.unrecognized.weight"},
        ):
            with self.assertRaisesRegex(RuntimeError, "Unmapped Qwen4Exp"):
                self.adapter.build_name_map(GGUFModelFiles(("fixture.gguf",)), config_for(text))

    def test_gdn_row_restore_and_conv_shape(self):
        prefix = "model.layers.0.linear_attn."
        qkv = torch.arange(20 * 5).reshape(20, 5).float()
        packed_qkv = torch.cat((qkv[:8], ggml_tile(qkv[8:], 0)), dim=0)
        gate = torch.arange(12 * 5).reshape(12, 5).float()
        alpha = torch.arange(6 * 5).reshape(6, 5).float()
        beta = alpha + 100
        conv = torch.arange(20 * 4).reshape(20, 4).float()
        packed_conv = torch.cat((conv[:8], ggml_tile(conv[8:], 0)), dim=0)
        output = self.run_weights([
            (prefix + "in_proj_qkv.weight", packed_qkv),
            (prefix + "in_proj_z.weight", ggml_tile(gate, 0)),
            (prefix + "in_proj_a.weight", ggml_tile(alpha, 0, head_dim=1)),
            (prefix + "in_proj_b.weight", ggml_tile(beta, 0, head_dim=1)),
            (prefix + "conv1d.weight", packed_conv),
        ])
        torch.testing.assert_close(output[prefix + "in_proj_qkv.weight"], qkv)
        torch.testing.assert_close(output[prefix + "in_proj_z.weight"], gate)
        torch.testing.assert_close(output[prefix + "in_proj_a.weight"], alpha)
        torch.testing.assert_close(output[prefix + "in_proj_b.weight"], beta)
        torch.testing.assert_close(output[prefix + "conv1d.weight"], conv.unsqueeze(1))

    def test_negative_exp_A_log_and_dt_bias_inverse(self):
        prefix = "model.layers.0.linear_attn."
        a = torch.linspace(-2, 2, 6)
        dt = torch.linspace(1, 3, 6)
        output = self.run_weights([
            (prefix + "A_log", ggml_tile(-a.exp(), 0, head_dim=1)),
            (prefix + "dt_bias", ggml_tile(dt, 0, head_dim=1)),
        ])
        torch.testing.assert_close(output[prefix + "A_log"], a)
        torch.testing.assert_close(output[prefix + "dt_bias"], dt)
        with self.assertRaisesRegex(ValueError, "negative exp"):
            self.run_weights([(prefix + "A_log", torch.ones(6))])

    def test_quantized_gdn_rows_keep_packed_columns(self):
        prefix = "model.layers.0.linear_attn."
        qkv = torch.arange(20 * 17).reshape(20, 17).to(torch.uint8)
        raw = torch.cat((qkv[:8], ggml_tile(qkv[8:], 0)), dim=0)
        out_packed = torch.arange(7 * 17).reshape(7, 17).to(torch.uint8)
        output = self.run_weights([
            (prefix + "in_proj_qkv.weight_type", torch.tensor(int(gguf.GGMLQuantizationType.Q8_0))),
            (prefix + "in_proj_qkv.weight", raw),
            (prefix + "out_proj.weight_type", torch.tensor(int(gguf.GGMLQuantizationType.Q8_0))),
            (prefix + "out_proj.weight", out_packed),
        ])
        self.assertTrue(torch.equal(output[prefix + "in_proj_qkv.weight"], qkv))
        self.assertTrue(torch.equal(output[prefix + "out_proj.weight"], out_packed))
        # The runtime input permutation, rather than corrupting packed columns,
        # must produce the same matmul as the original grouped-value weights.
        native_w = torch.arange(7 * 12).reshape(7, 12).float()
        stored_w = ggml_tile(native_w, 1)
        x = torch.arange(3 * 12).reshape(3, 12).float()
        layout = self.adapter.get_linear_layouts(
            GGUFModelFiles(("fixture.gguf",)), self.config,
            {"ssm_out": prefix + "out_proj.weight"},
        )[prefix + "out_proj"]
        torch.testing.assert_close(layout.input_to_gguf(x) @ stored_w.T, x @ native_w.T)
        dense_output = self.run_weights([(prefix + "out_proj.weight", stored_w)])
        torch.testing.assert_close(dense_output[prefix + "out_proj.weight"], native_w)

    def test_zero_centered_norms_are_restored_but_gdn_norm_is_unchanged(self):
        norm_names = [
            "model.layers.0.attn_hyper_connection.hc_norm.weight",
            "model.hyper_connection_mixer.hc_norm.weight",
            "model.layers.1.ple.norm_key.weight",
            "model.layers.1.ple.norm_query.weight",
            "model.layers.1.ple.norm_conv.weight",
            "model.layers.3.self_attn.q_norm.weight",
            "model.layers.3.self_attn.k_norm.weight",
            "model.layers.3.self_attn.indexer.q_layernorm.weight",
            "model.layers.3.self_attn.indexer.k_layernorm.weight",
        ]
        gamma = torch.linspace(-0.5, 0.5, 8)
        gdn = "model.layers.0.linear_attn.norm.weight"
        output = self.run_weights([(name, gamma + 1) for name in norm_names] + [(gdn, gamma)])
        for name in norm_names:
            torch.testing.assert_close(output[name], gamma)
        self.assertTrue(torch.equal(output[gdn], gamma))

    def test_HC_Q8_cpu_decode_has_no_weight_type_leak(self):
        # Construct GGML Q8_0 blocks independently: fp16 scale +32int8 codes.
        codes = np.arange(-32, 32, dtype=np.int8).reshape(2, 32)
        scales = np.array([0.125, 0.5], dtype=np.float16)
        packed = np.concatenate((scales.view(np.uint8).reshape(2, 2), codes.view(np.uint8)), axis=1)
        name = "model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
        type_name = name.removesuffix(".weight") + ".weight_type"
        output = self.run_weights([
            (type_name, torch.tensor(int(gguf.GGMLQuantizationType.Q8_0))),
            (name, torch.from_numpy(packed)),
        ])
        self.assertEqual(set(output), {name})
        expected = torch.from_numpy(codes.astype(np.float32) * scales.astype(np.float32)[:, None]).bfloat16()
        torch.testing.assert_close(output[name], expected, rtol=0, atol=0)

    def test_indexer_join_is_Q_then_K_and_incomplete_halves_fail(self):
        prefix = "model.layers.3.self_attn.indexer."
        q, k = torch.randn(4, 8).bfloat16(), torch.randn(2, 8).bfloat16()
        output = self.run_weights([(prefix + "k_proj.weight", k), (prefix + "q_proj.weight", q)])
        joined = output[prefix + "index_qk_proj.weight"]
        self.assertTrue(torch.equal(joined, torch.cat([q, k])))
        x = torch.randn(3, 8)
        torch.testing.assert_close(x @ joined.float().T, torch.cat([x @ q.float().T, x @ k.float().T], dim=1))
        with self.assertRaisesRegex(ValueError, "Missing Qwen4Exp indexer"):
            self.run_weights([(prefix + "q_proj.weight", q)])

    def test_QSA_query_and_gate_rows_keep_checkpoint_head_order(self):
        # Native Qwen3Next/QSA splits each Q-head's [Q,gate] rows after GEMM.
        # The Qwen4 GGUF converter preserves this Q projection row order.
        heads, dim, hidden = 3, 2, 5
        q_gate = torch.arange(heads * 2 * dim * hidden).reshape(heads * 2 * dim, hidden).float()
        name = "model.layers.3.self_attn.q_proj.weight"
        output = self.run_weights([(name, q_gate)])
        x = torch.randn(2, hidden)
        flat = x @ output[name].T
        q, gate = flat.reshape(2, heads, 2 * dim).chunk(2, dim=-1)
        expected_q = torch.stack([x @ q_gate[h * 2 * dim:h * 2 * dim + dim].T for h in range(heads)], dim=1)
        expected_gate = torch.stack([x @ q_gate[h * 2 * dim + dim:(h + 1) * 2 * dim].T for h in range(heads)], dim=1)
        torch.testing.assert_close(q, expected_q)
        torch.testing.assert_close(gate, expected_gate)

    def test_expert_data_split_preserves_bytes_and_type(self):
        base = "model.layers.0.mlp.experts.0.gate_proj"
        packed = torch.arange(2 * 4 * 17).reshape(2, 4, 17).to(torch.uint8)
        kind = torch.tensor(int(gguf.GGMLQuantizationType.IQ3_S))
        output = self.run_weights([(base + ".weight_type", kind), (base + ".weight", packed)])
        self.assertTrue(torch.equal(output[base + ".weight_type"], kind))
        self.assertTrue(torch.equal(output[base + ".weight"], packed[0]))
        self.assertTrue(torch.equal(output[base.replace("experts.0", "experts.1") + ".weight"], packed[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory")
    parser.add_argument("--config")
    _ARGS, unittest_args = parser.parse_known_args()
    unittest.main(argv=[__file__, *unittest_args], verbosity=2)
