from typing import List
from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.decoders.rnn import RNNDecoder
from espnet_onnx.asr.model.decoders.xformer import XformerDecoder
from espnet_onnx.asr.model.decoders.transducer import TransducerDecoder


def get_decoder(config: Config, providers: List[str], use_quantized: bool = False,
                optimize_option = None):
    if config.dec_type == 'RNNDecoder':
        return RNNDecoder(config, providers, use_quantized, optimize_option)
    elif config.dec_type == 'TransducerDecoder':
        return TransducerDecoder(config, providers, use_quantized, optimize_option)
    else:
        return XformerDecoder(config, providers, use_quantized, optimize_option)
