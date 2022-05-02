# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.
#
# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to check optimizations from OnnxRuntime.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
#
# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16 for mixed precision inference in GPU with Tensor Core.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.

# This script is modified to optimize ESPnet.
# Copyright (c) 2022 Masao Someki.
# Licensed under the MIT License.

import argparse
import logging
import os
from typing import Dict, Optional

from onnx import ModelProto, load_model

from .fusion_options import FusionOptions
from .transformer_encoder import TransformerEncoderOnnxModel

logger = logging.getLogger(__name__)

MODEL_TYPES = {
    "TransformerEncoder": (TransformerEncoderOnnxModel, "pytorch", 1),
    # "TransformerDecoder": (BartOnnxModel, "pytorch", 1),
    # "TransformerLM": (BartOnnxModel, "pytorch", 1),
}


def optimize_by_onnxruntime(
    onnx_model_path: str,
    use_gpu: bool = False,
    optimized_model_path: Optional[str] = None,
    opt_level: Optional[int] = 99,
    disabled_optimizers=[],
) -> str:
    """
    Use onnxruntime to optimize model.
    Args:
        onnx_model_path (str): the path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
        opt_level (int): graph optimization level.
        disabled_optimizers (List[str]): a list of names of disabled optimizers
    Returns:
        optimized_model_path (str): the path of optimized model
    """
    assert opt_level in [1, 2, 99]
    import onnxruntime

    if use_gpu and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    if opt_level == 1:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt_level == 2:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  # remove .onnx suffix
        optimized_model_path = "{}_o{}_{}.onnx".format(path_prefix, opt_level, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    kwargs = {}
    if disabled_optimizers:
        kwargs["disabled_optimizers"] = disabled_optimizers

    if not use_gpu:
        session = onnxruntime.InferenceSession(
            onnx_model_path, sess_options, providers=["CPUExecutionProvider"], **kwargs
        )
    else:
        session = onnxruntime.InferenceSession(
            onnx_model_path, sess_options, providers=["CUDAExecutionProvider"], **kwargs
        )
        assert "CUDAExecutionProvider" in session.get_providers()  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.debug("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path


def optimize_by_fusion(
    model: ModelProto,
    model_type: str,
    num_heads: int,
    hidden_size: int,
    optimization_options: Optional[FusionOptions] = None,
):
    """Optimize Model by graph fusion logic.
    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.
    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.
    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.
     Returns:
        object of an optimizer class.
    """
    if model_type is None:
        raise ValueError('Please specify model_type from ["TransformerEncoder", "TransformerDecoder", "TransformerLM"]')

    (optimizer_class, producer, _) = MODEL_TYPES[model_type]

    if model.producer_name and producer != model.producer_name:
        logger.warning(
            f"Model producer not matched: Expect {producer}, Got {model.producer_name} {model.producer_version}. Please specify correct --model_type parameter."
        )

    if optimization_options is None:
        optimization_options = FusionOptions(model_type)

    optimizer = optimizer_class(model, num_heads, hidden_size)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    optimizer.model.producer_name = "onnxruntime.transformer"
    from onnxruntime import __version__ as onnxruntime_version

    optimizer.model.producer_version = onnxruntime_version

    return optimizer


def optimize_model(
    input: str,
    model_type: str = "bert",
    num_heads: int = 0,
    hidden_size: int = 0,
    optimization_options: Optional[FusionOptions] = None,
    opt_level: int = None,
    use_gpu: bool = False,
    only_onnxruntime: bool = False,
):
    """Optimize Model by OnnxRuntime and/or python fusion logic.
    ONNX Runtime has graph optimizations (https://onnxruntime.ai/docs/resources/graph-optimizations.html).
    However, the coverage is limited. We also have graph fusions that implemented in Python to improve the coverage.
    They can combined: ONNX Runtime will run first when opt_level > 0, then graph fusions in Python will be applied.
    To use ONNX Runtime only and no Python fusion logic, use only_onnxruntime flag and a positive opt_level like
        optimize_model(input, opt_level=1, use_gpu=False, only_onnxruntime=True)
    When opt_level is None, we will choose default optimization level according to model type.
    When opt_level is 0 and only_onnxruntime is False, only python fusion logic is used and onnxruntime is disabled.
    When opt_level > 1, use_gpu shall set properly since the optimized graph might contain operators for GPU or CPU only.
    If your model is intended for GPU inference only (especially float16 or mixed precision model), it is recommended to
    set use_gpu to be True, otherwise the model is not optimized for GPU inference.
    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.
    Args:
        input (str): input model path.
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.
        opt_level (int, optional): onnxruntime graph optimization level (0, 1, 2 or 99) or None. Defaults to None.
                                   When the value is None, default value (1 for bert and gpt2, 0 for other model types) will be used.
                                   When the level > 0, onnxruntime will be used to optimize model first.
        use_gpu (bool, optional): use gpu or not for onnxruntime. Defaults to False.
        only_onnxruntime (bool, optional): only use onnxruntime to optimize model, and no python fusion. Defaults to False.
     Returns:
        object of an optimizer class.
    """
    assert opt_level is None or opt_level in [0, 1, 2, 99]

    if model_type != "bert" and (num_heads == 0 or hidden_size == 0):
        logger.warning("Please specify parameters of num_heads and hidden_size when model_type is not 'bert'")

    (optimizer_class, producer, default_opt_level) = MODEL_TYPES[model_type]

    if opt_level is None:
        opt_level = default_opt_level

    temp_model_path = None
    if opt_level > 1:
        # Disable some optimizers that might cause failure in symbolic shape inference or attention fusion.
        disabled_optimizers = (
            []
            if only_onnxruntime
            else [
                "MatMulScaleFusion",
                "MatMulAddFusion", "SimplifiedLayerNormFusion",
                "GemmActivationFusion",
                "BiasSoftmaxFusion",
            ]
        )
        temp_model_path = optimize_by_onnxruntime(
            input,
            use_gpu=use_gpu,
            opt_level=opt_level,
            disabled_optimizers=disabled_optimizers,
        )
    elif opt_level == 1:
        # basic optimizations (like constant folding and cast elimation) are not specified to exection provider.
        # CPU provider is used here so that there is no extra node for GPU memory copy.
        temp_model_path = optimize_by_onnxruntime(input, use_gpu=False, opt_level=1)

    if only_onnxruntime and not temp_model_path:
        logger.warning("Please specify a positive value for opt_level when only_onnxruntime is True")

    model = load_model(temp_model_path or input)

    if only_onnxruntime:
        optimizer = optimizer_class(model, num_heads, hidden_size)
    else:
        optimizer = optimize_by_fusion(model, model_type, num_heads, hidden_size, optimization_options)

    # Remove the temporary model.
    if temp_model_path:
        os.remove(temp_model_path)
        logger.debug("Remove tempoary model: {}".format(temp_model_path))

    return optimizer
