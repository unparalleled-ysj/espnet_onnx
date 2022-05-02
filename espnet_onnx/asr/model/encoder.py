from typing import List
from espnet_onnx.utils.config import Config
from espnet_onnx.asr.model.encoders.encoder import Encoder
from espnet_onnx.asr.model.encoders.streaming import StreamingEncoder


def get_encoder(config: Config, providers: List[str], use_quantized: bool = False,
                optimize_option = None):
    if config.enc_type == 'ContextualXformerEncoder':
        return StreamingEncoder(config, providers, use_quantized, optimize_option)
    else:
        return Encoder(config, providers, use_quantized, optimize_option)
