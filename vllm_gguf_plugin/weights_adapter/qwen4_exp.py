# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp text GGUF -> native vLLM AMD checkpoint adapter.

The large PLE table is intentionally excluded from the ordinary name map.  The
loader must append ``iter_qwen4_ple_weights`` from ``qwen4_ple``; it produces
native BF16 checkpoint row shards and the exact hash buffers from GGUF metadata.
The other GGUF matrices, including mixed-quant expert tensors, remain packed.

Inverses below follow llama.cpp's Qwen4Exp/Qwen3Next conversion: tiled GDN
value heads, negative exponentiated A_log, and Gemma's zero-centred norm gamma.
HC projections are decoded on CPU because the native AMD HC modules explicitly
use quant_config=None.  QSA's separately stored BF16 indexer Q/K are rejoined.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import gguf
import torch

from ..gguf_files import GGUFModelFiles
from ..quantization.layout import GGUFHeadTilingLayout, GGUFLinearLayout
from ..weight_utils import get_gguf_tensor_names, split_stacked_experts
from .base import BaseGGUFWeightsAdapter, GGUFWeight

if TYPE_CHECKING:
    from transformers import PretrainedConfig
    from vllm.config import ModelConfig


QWEN4_MODEL_TYPES = ("qwen4_exp", "qwen4_exp_text")
PLE_GGUF_NAME = "per_layer_token_embd.weight"

_GLOBAL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_hc_norm.weight": "model.hyper_connection_mixer.hc_norm.weight",
    "output_hc_down.weight": (
        "model.hyper_connection_mixer.input_mix_weight_down.weight"
    ),
    "output_hc_up.weight": "model.hyper_connection_mixer.input_mix_weight_up.weight",
}

_LAYER_NAMES = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_a.weight": "linear_attn.A_log",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_exps.weight": "mlp.experts.0.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.0.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.0.down_proj.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "hc_attn_norm.weight": "attn_hyper_connection.hc_norm.weight",
    "hc_attn_down.weight": "attn_hyper_connection.input_mix_weight_down.weight",
    "hc_attn_up.weight": "attn_hyper_connection.input_mix_weight_up.weight",
    "hc_attn_inject.weight": "attn_hyper_connection.block_inject_weight.weight",
    "hc_ffn_norm.weight": "mlp_hyper_connection.hc_norm.weight",
    "hc_ffn_down.weight": "mlp_hyper_connection.input_mix_weight_down.weight",
    "hc_ffn_up.weight": "mlp_hyper_connection.input_mix_weight_up.weight",
    "hc_ffn_inject.weight": "mlp_hyper_connection.block_inject_weight.weight",
    "ple_key.weight": "ple.key_proj.weight",
    "ple_value.weight": "ple.value_proj.weight",
    "ple_norm_key.weight": "ple.norm_key.weight",
    "ple_norm_query.weight": "ple.norm_query.weight",
    "ple_norm_conv.weight": "ple.norm_conv.weight",
    "ple_conv1d.weight": "ple.conv1d.weight",
    "indexer.q_proj.weight": "self_attn.indexer.q_proj.weight",
    "indexer.k_proj.weight": "self_attn.indexer.k_proj.weight",
    "indexer.q_norm.weight": "self_attn.indexer.q_layernorm.weight",
    "indexer.k_norm.weight": "self_attn.indexer.k_layernorm.weight",
}


def qwen4_text_config(model_config):
    config = model_config.hf_config
    return config.get_text_config()


def qwen4_ple_prefix(model_config) -> str:
    """Checkpoint prefix consumed by the native AMD PLE child loader."""
    layers = qwen4_text_config(model_config).ple_layer_ids
    if len(layers) != 1 or int(layers[0]) <= 0:
        raise ValueError("This GGUF adapter requires exactly one 1-based PLE layer")
    return f"model.layers.{int(layers[0]) - 1}.ple.ple_embedding"


def _gdn_layout(text_config) -> GGUFHeadTilingLayout | None:
    key_heads = int(text_config.linear_num_key_heads)
    value_heads = int(text_config.linear_num_value_heads)
    repeat, remainder = divmod(value_heads, key_heads)
    if remainder:
        raise ValueError("GDN value heads must be divisible by key heads")
    if repeat <= 1:
        return None
    return GGUFHeadTilingLayout(
        heads_per_group=repeat,
        head_dim=int(text_config.linear_value_head_dim),
    )


def expected_qwen4_tensor_names(text_config) -> set[str]:
    """Required text checkpoint coverage, before any weight values are read."""
    names = set(_GLOBAL_NAMES) | {PLE_GGUF_NAME}
    layer_types = text_config.layer_types
    if len(layer_types) != int(text_config.num_hidden_layers):
        raise ValueError("Qwen4Exp layer_types length does not match num_hidden_layers")
    ple_layers = {int(layer) - 1 for layer in text_config.ple_layer_ids}
    for index, layer_type in enumerate(layer_types):
        suffixes = {name for name in _LAYER_NAMES if name.startswith(("ffn_", "hc_"))}
        if layer_type == "linear_attention":
            suffixes |= {
                name
                for name in _LAYER_NAMES
                if name.startswith("ssm_") and name != "ssm_a.weight"
            } | {"attn_qkv.weight", "attn_gate.weight"}
        elif layer_type in (
            "full_attention",
            "indexed_attention",
            "qwen_sparse_attention",
        ):
            suffixes |= {
                name
                for name in _LAYER_NAMES
                if name.startswith(("attn_", "indexer."))
                and name not in ("attn_qkv.weight", "attn_gate.weight")
            }
        else:
            raise ValueError(f"Unsupported Qwen4Exp layer type: {layer_type}")
        if index in ple_layers:
            suffixes |= {name for name in _LAYER_NAMES if name.startswith("ple_")}
        names.update(f"blk.{index}.{suffix}" for suffix in suffixes)
    return names


def _is_hc_projection(name: str) -> bool:
    stem = name.removesuffix(".weight_type").removesuffix(".weight")
    return (
        "_hyper_connection." in stem or ".hyper_connection_mixer." in stem
    ) and stem.endswith(
        (".input_mix_weight_down", ".input_mix_weight_up", ".block_inject_weight")
    )


def _dequantize_cpu(weight: torch.Tensor, quant_type: int) -> torch.Tensor:
    """Decode only small HC matrices; never the PLE table or expert tensors."""
    if weight.device.type != "cpu":
        raise ValueError("Qwen4Exp GGUF HC dequantization must run on CPU")
    decoded = gguf.quants.dequantize(
        weight.contiguous().numpy(), gguf.GGMLQuantizationType(quant_type)
    )
    return torch.from_numpy(decoded).to(dtype=torch.bfloat16)


class Qwen4ExpGGUFAdapter(BaseGGUFWeightsAdapter):
    """Explicit, strict text-only adapter for the Qwen3.8 Flash Next GGUF."""

    # Both indexer source halves are BF16 in the verified Huihui checkpoint.
    # They are concatenated before loading into this native (unfused) module.
    extra_unquantized_modules = ("index_qk_proj",)

    @classmethod
    def matches(cls, config) -> bool:
        return config.model_type in QWEN4_MODEL_TYPES

    @classmethod
    def architecture(cls, config) -> str | None:
        del config
        return "Qwen4ExpForCausalLM"

    @staticmethod
    def map_name(name: str) -> str | None:
        if name == PLE_GGUF_NAME:
            return None
        if name in _GLOBAL_NAMES:
            return _GLOBAL_NAMES[name]
        match = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if match is not None and match[2] in _LAYER_NAMES:
            return f"model.layers.{match[1]}.{_LAYER_NAMES[match[2]]}"
        return None

    def patch_hf_config(
        self, files: GGUFModelFiles, hf_config: PretrainedConfig
    ) -> PretrainedConfig:
        if files.mm_proj is not None:
            raise NotImplementedError(
                "Qwen4Exp GGUF adapter currently supports text only"
            )
        text = hf_config.get_text_config()
        # The checkpoint names full attention 'full_attention'; recent vLLM
        # names Qwen4Exp sparse full-attention layers 'indexed_attention'.
        if hasattr(text, "layer_types"):
            text.layer_types = [
                "indexed_attention" if value == "full_attention" else value
                for value in text.layer_types
            ]
        # The PLE sidecar is decoded BF16.  Do not inherit FP8/FP4 PLE metadata
        # from a config belonging to a different weight checkpoint.
        text.ple_embedding_dtype = None
        hf_config.architectures = ["Qwen4ExpForCausalLM"]
        if text is not hf_config:
            text.architectures = ["Qwen4ExpForCausalLM"]
        # Shard one can be metadata-only.  Scan all headers for the vocabulary.
        for path in files.backbone:
            for tensor in gguf.GGUFReader(path).tensors:
                if tensor.name == "token_embd.weight":
                    text.vocab_size = int(tensor.shape[1])
                    if hasattr(hf_config, "vocab_size"):
                        hf_config.vocab_size = text.vocab_size
        return hf_config

    def build_name_map(
        self, files: GGUFModelFiles, model_config: ModelConfig
    ) -> dict[str, str]:
        names = get_gguf_tensor_names(files.all_files)
        expected = expected_qwen4_tensor_names(qwen4_text_config(model_config))
        # Some GGUF writers append .weight to the state-space A tensor.
        missing = expected - names
        missing = {
            name
            for name in missing
            if not (name.endswith(".ssm_a") and name + ".weight" in names)
        }
        if missing:
            raise RuntimeError(f"Missing Qwen4Exp GGUF tensors: {sorted(missing)}")
        mapped: dict[str, str] = {}
        unknown: list[str] = []
        for name in sorted(names):
            target = self.map_name(name)
            if target is not None:
                mapped[name] = target
            elif name != PLE_GGUF_NAME:
                unknown.append(name)
        if unknown:
            raise RuntimeError(f"Unmapped Qwen4Exp GGUF tensors: {unknown}")
        if len(set(mapped.values())) != len(mapped):
            raise RuntimeError("Qwen4Exp GGUF mapping has duplicate destinations")
        return mapped

    def get_linear_layouts(
        self,
        files: GGUFModelFiles,
        model_config: ModelConfig,
        name_map: dict[str, str],
    ) -> dict[str, GGUFLinearLayout]:
        del files
        layout = _gdn_layout(qwen4_text_config(model_config))
        if layout is None:
            return {}
        return {
            name.removesuffix(".weight"): layout
            for name in name_map.values()
            if name.endswith("linear_attn.out_proj.weight")
        }

    @staticmethod
    def _restore_gdn(
        name: str,
        weight: torch.Tensor,
        text_config,
        layout: GGUFHeadTilingLayout | None,
        quantized_bases: set[str],
    ) -> torch.Tensor:
        base = name.removesuffix(".weight")
        if base.endswith(".A_log"):
            if not torch.all(weight < 0):
                raise ValueError(f"GGUF {name} must contain negative exp(A_log)")
            weight = torch.log(-weight)
        if layout is None or "linear_attn." not in name:
            return weight
        if base.endswith("linear_attn.out_proj"):
            if base not in quantized_bases:
                return layout.weight_to_vllm(weight, dim=1)
        elif base.endswith((".A_log", ".dt_bias")):
            return layout.weight_to_vllm(weight, dim=0, head_dim=1)
        elif base.endswith("linear_attn.in_proj_z"):
            return layout.weight_to_vllm(weight, dim=0)
        elif base.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
            return layout.weight_to_vllm(weight, dim=0, head_dim=1)
        elif base.endswith(("linear_attn.in_proj_qkv", "linear_attn.conv1d")):
            qk_rows = (
                int(text_config.linear_key_head_dim)
                * int(text_config.linear_num_key_heads)
                * 2
            )
            value = layout.weight_to_vllm(weight[qk_rows:], dim=0)
            return torch.cat([weight[:qk_rows], value], dim=0)
        return weight

    def transform_weights(
        self, weights: Iterable[GGUFWeight], model_config: ModelConfig
    ) -> Iterable[GGUFWeight]:
        text = qwen4_text_config(model_config)
        layout = _gdn_layout(text)

        def transformed() -> Iterable[GGUFWeight]:
            quantized_bases: set[str] = set()
            hc_types: dict[str, int] = {}
            index_parts: dict[str, dict[str, torch.Tensor]] = {}
            for name, weight in weights:
                if name.endswith(".weight_type"):
                    base = name.removesuffix(".weight_type")
                    if _is_hc_projection(name):
                        hc_types[base] = int(weight.item())
                        continue
                    if ".indexer." in name:
                        raise NotImplementedError(
                            "Qwen4Exp indexer adapter requires BF16 Q/K source weights"
                        )
                    quantized_bases.add(base)
                    yield name, weight
                    continue
                base = name.removesuffix(".weight")
                if base in hc_types:
                    weight = _dequantize_cpu(weight, hc_types.pop(base))
                # Keep only the two small indexer halves, regardless of source
                # shard/tensor order, and fail if either half is absent.
                index = re.fullmatch(r"(.+\.indexer)\.([qk])_proj\.weight", name)
                if index is not None:
                    parts = index_parts.setdefault(index[1], {})
                    if index[2] in parts:
                        raise ValueError(f"Duplicate indexer tensor: {name}")
                    parts[index[2]] = weight
                    if len(parts) == 2:
                        q, k = parts["q"], parts["k"]
                        if q.ndim != 2 or k.ndim != 2 or q.shape[1] != k.shape[1]:
                            raise ValueError(
                                f"Invalid Qwen4Exp indexer Q/K shapes: {name}"
                            )
                        yield (
                            f"{index[1]}.index_qk_proj.weight",
                            torch.cat([q, k], dim=0),
                        )
                        del index_parts[index[1]]
                    continue
                weight = self._restore_gdn(name, weight, text, layout, quantized_bases)
                if (
                    name.endswith("norm.weight")
                    and not name.endswith("linear_attn.norm.weight")
                ) or name.endswith(
                    (
                        ".ple.norm_key.weight",
                        ".ple.norm_query.weight",
                        ".ple.norm_conv.weight",
                    )
                ):
                    weight = weight - 1
                if name.endswith("conv1d.weight") and weight.ndim == 2:
                    weight = weight.unsqueeze(1)
                elif (
                    name.endswith(".weight")
                    and weight.ndim == 1
                    and "norm" not in name
                    and base not in quantized_bases
                ):
                    weight = weight.unsqueeze(0)
                yield name, weight
            if hc_types:
                raise ValueError(
                    f"Missing packed Qwen4Exp HC weights: {sorted(hc_types)}"
                )
            if index_parts:
                raise ValueError(
                    f"Missing Qwen4Exp indexer Q/K half: {sorted(index_parts)}"
                )

        yield from split_stacked_experts(transformed())
