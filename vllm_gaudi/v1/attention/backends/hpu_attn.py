# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

from dataclasses import dataclass
from typing import Optional

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm_gaudi.attention.backends.hpu_attn import (HPUAttentionBackend,
                                                    HPUAttentionMetadata)
from vllm_gaudi.extension.logger import logger as init_logger

logger = init_logger()


class HPUAttentionBackendV1(HPUAttentionBackend):

    @staticmethod
    def get_builder_cls() -> type["HPUAttentionMetadataV1Builder"]:
        return HPUAttentionMetadataV1Builder
    
    @staticmethod
    def get_name() -> str:
        return "HPU_ATTN_V1"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return HPUAttentionMetadataV1


@dataclass
class HPUAttentionMetadataV1(HPUAttentionMetadata):
    # TODO(kwisniewski98): for now, in V1 input positions are not provided
    # which needs to be fixed in the future, as we need to support MLA
    """Metadata for HPUAttentionbackend."""
    is_prompt: bool
    attn_bias: Optional[torch.Tensor]

    seq_lens_tensor: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]

    @classmethod
    def make_prefill_metadata(cls, attn_bias, block_list, context_lens_tensor,
                              seq_lens_tensor, slot_mapping, block_size):
        return cls(
            is_prompt=True,
            block_list=block_list,
            block_mapping=None,
            block_usage=None,
            block_groups=None,
            attn_bias=attn_bias,
            alibi_blocks=None,
            num_decode_tokens=0,
            context_lens_tensor=context_lens_tensor,
            seq_lens_tensor=seq_lens_tensor,
            multi_modal_placeholder_index_maps=None,
            num_prefills=0,  # ignored on HPU
            num_prefill_tokens=0,  # ignored on HPU
            input_positions=None,
            slot_mapping=slot_mapping,
            enable_kv_scales_calculation=False,
            block_size=block_size)

    @classmethod
    def make_decode_metadata(cls, block_list, block_usage, block_groups,
                             input_positions, num_decode_tokens, slot_mapping,
                             block_size):
        return cls(
            is_prompt=False,
            block_mapping=None,
            alibi_blocks=None,
            attn_bias=None,
            seq_lens_tensor=None,
            context_lens_tensor=None,
            num_prefills=0,  # ignored on HPU
            num_prefill_tokens=0,  # ignored on HPU
            multi_modal_placeholder_index_maps=None,
            block_list=block_list,
            block_usage=block_usage,
            block_groups=block_groups,
            input_positions=input_positions,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping,
            enable_kv_scales_calculation=False,
            block_size=block_size)

@dataclass
class HPUCommonAttentionMetadata:
    """
    Per-batch attention metadata, shared across layers and backends.
    AttentionMetadataBuilder instances use it to construct per-layer metadata.
    
    For many of the tensors we keep both GPU and CPU versions.
    """
    # copied from CommonAttentionMetadata property as none

    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc_cpu: Optional[torch.Tensor] = None
    """(batch_size + 1,), the start location of each request in query Tensor"""

    seq_lens: Optional[torch.Tensor] = None
    seq_lens_cpu: Optional[torch.Tensor] = None
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""

    num_computed_tokens_cpu: Optional[torch.Tensor] = None
    """(batch_size,), the number of computed tokens for each request"""

    num_reqs: Optional[int] = None
    """Number of requests"""
    num_actual_tokens: Optional[int] = None
    """Total number of tokens in batch"""
    max_query_len: Optional[int] = None
    """Longest query in batch"""
    max_seq_len: Optional[int] = None
    """Longest context length in batch"""

    block_table_tensor: Optional[torch.Tensor] = None

    causal: bool = True
 
    hpu_attn_metadata: Optional[HPUAttentionMetadataV1] = None


class HPUAttentionMetadataV1Builder:

    def __init__(self, layer_names: list[str]):
        pass

    def build(self,
              common_attn_metadata: HPUCommonAttentionMetadata) -> HPUAttentionMetadataV1:
        attn_metadata = common_attn_metadata.hpu_attn_metadata
        return HPUCommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            slot_mapping=common_attn_metadata.slot_mapping,
            hpu_attn_metadata=HPUAttentionMetadataV1.make_decode_metadata(
                block_list=attn_metadata.block_list,
                block_usage=attn_metadata.block_usage,
                block_groups=attn_metadata.block_groups,
                input_positions=None,
                num_decode_tokens=attn_metadata.num_decode_tokens,
                slot_mapping=attn_metadata.slot_mapping,
                block_size=attn_metadata.block_size,
            )
        )

    def build_for_drafting(
        self,
        common_attn_metadata: HPUCommonAttentionMetadata,
        draft_index: int,
    ):
        """
        Build attention metadata for draft model. Uses build by default.
        
        Args:
            common_attn_metadata: The common attention metadata.
            draft_index: The index of the current draft operation.
                When speculating a chain of tokens, this index refers to the
                draft attempt for the i-th token.
                For tree-based attention, this index instead refers to the
                draft attempt for the i-th level in the tree of tokens.
        """
        return self.build(common_attn_metadata=common_attn_metadata)