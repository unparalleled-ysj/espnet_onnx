from typing import List

import onnxruntime


class JointNetwork:
    def __init__(
        self,
        config,
        providers: List[str],
        use_quantized=False,
        optimize_option: onnxruntime.SessionOptions = None,
    ):
        if use_quantized:
            self.joint_session = onnxruntime.InferenceSession(
                config.quantized_model_path,
                sess_options=optimize_option,
                providers=providers
            )
        else:
            self.joint_session = onnxruntime.InferenceSession(
                config.model_path,
                sess_options=optimize_option,
                providers=providers
            )

    def __call__(self, enc_out, dec_out):
        input_dict = {
            'enc_out': enc_out,
            'dec_out': dec_out
        }
        return self.joint_session.run(['joint_out'], input_dict)[0]
