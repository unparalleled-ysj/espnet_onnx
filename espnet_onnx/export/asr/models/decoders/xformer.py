import os

import torch
import torch.nn as nn
import numpy as np

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from ..multihead_att import OnnxMultiHeadedAttention

from espnet_onnx.utils.function import subsequent_mask
from ..language_models.embed import Embedding
from espnet_onnx.utils.abs_model import AbsExportModel


class XformerDecoder(nn.Module, AbsExportModel):
    def __init__(self, model, max_seq_len=512, **kwargs):
        super().__init__()
        self.embed = Embedding(model.embed, max_seq_len)
        self.model = model
        # replace multihead attention module into customized module.
        for d in self.model.decoders:
            # d is DecoderLayer
            if isinstance(d.self_attn, MultiHeadedAttention):
                d.self_attn = OnnxMultiHeadedAttention(d.self_attn)
            if isinstance(d.src_attn, MultiHeadedAttention):
                d.src_attn = OnnxMultiHeadedAttention(d.src_attn)
                
        self.num_heads = model.decoders[0].self_attn.h
        self.hidden_size = model.decoders[0].self_attn.linear_out.out_features

    def forward(self, tgt, tgt_mask, memory, cache):
        x = self.embed(tgt)
        new_cache = []
        for c, decoder in zip(cache, self.model.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)  # (1, L, 512) * n_layer
        y = self.model.after_norm(x[:, -1])
        y = torch.log_softmax(self.model.output_layer(y), dim=-1)
        return y, new_cache

    def get_dummy_inputs(self, enc_size):
        tgt = torch.LongTensor([0, 1]).unsqueeze(0)
        tgt_mask = torch.from_numpy(subsequent_mask(2)[None, :]).float()
        enc_out = torch.randn(1, 100, enc_size)
        cache = [
            torch.zeros((1, 1, self.model.decoders[0].size))
            for _ in range(len(self.model.decoders))
        ]
        return (tgt, tgt_mask, enc_out, cache)

    def is_optimizable(self):
        return True

    def get_input_names(self):
        return ['tgt', 'tgt_mask', 'memory'] \
            + ['cache_%d' % i for i in range(len(self.model.decoders))]

    def get_output_names(self):
        return ['y'] \
            + ['out_cache_%d' % i for i in range(len(self.model.decoders))]

    def get_dynamic_axes(self):
        ret = {
            'tgt': {
                0: 'tgt_batch',
                1: 'tgt_length'
            },
            'tgt_mask': {
                1: 'tgt_mask_length',
                2: 'tgt_mask_height'
            },
            'memory': {
                0: 'memory_batch',
                1: 'memory_length'
            }
        }
        ret.update({
            'cache_%d' % d: {
                0: 'cache_%d_batch' % d,
                1: 'cache_%d_length' % d
            }
            for d in range(len(self.model.decoders))
        })
        return ret

    def get_model_config(self, path):
        file_name = os.path.join(path, 'decoder.onnx')
        return {
            "dec_type": "XformerDecoder",
            "model_path": file_name,
            "n_layers": len(self.model.decoders),
            "odim": self.model.decoders[0].size
        }
