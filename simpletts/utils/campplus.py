import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi


class SpkEmbExtractor:

    def __init__(self, model_path, intra_op_num_threads=1, providers=None):
        # campplus.onnx
        self.model_path = model_path
        self.intra_op_num_threads = intra_op_num_threads
        self.providers = providers or ["CPUExecutionProvider"]
        self.ort_session = self._create_session()

    def _create_session(self):
        """Create an ONNX Runtime session with specified options."""
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = self.intra_op_num_threads
        return onnxruntime.InferenceSession(self.model_path,
                                            sess_options=options,
                                            providers=self.providers)

    def __getstate__(self):
        """Return the state for pickling."""
        state = self.__dict__.copy()
        state["ort_session"] = None
        return state

    def __setstate__(self, state):
        """Restore the state and recreate the ONNX session."""
        self.__dict__.update(state)
        self.ort_session = self._create_session()

    def __call__(self, waveform: torch.Tensor, sample_rate):
        """Run inference on the model."""
        if self.ort_session is None:
            raise RuntimeError("ONNX session is not initialized.")
        assert sample_rate == 16000
        feat = kaldi.fbank(waveform,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=sample_rate)

        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = self.ort_session.run(
            None, {
                self.ort_session.get_inputs()[0].name:
                feat.unsqueeze(dim=0).cpu().numpy()
            })[0].flatten().tolist()

        return torch.nn.functional.normalize(
            torch.tensor(embedding, dtype=torch.float32),
            dim=0,
        )
