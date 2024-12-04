from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class TTSLLMModelConfig:
    # TODO: fix vocab size
    vocab_size: int = 350000
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 16
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 1408
    shared_expert_intermediate_size: int = 5632
    num_experts_per_tok: int = 2
    num_experts: int = 8
    norm_topk_prob: bool = False
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: Optional[int] = None
    spk_embed_dim: Optional[int] = 192


@dataclass
class DITModelConfig:
    # TODO: fix vocab size
    n_mels: int = 80
    vocab_size: int = 8192 * 4
    frequency_embedding_size: int = 1024
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    num_key_value_heads: int = 4  # MGA
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    use_sliding_window: bool = False
    sliding_window: int = 4096
    max_window_layers: int = 28
    attention_dropout: float = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 1024
    shared_expert_intermediate_size: int = 2048
    num_experts_per_tok: int = 2
    num_experts: int = 8
    norm_topk_prob: bool = False
    output_router_logits: bool = True
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: Optional[int] = None
    spk_embed_dim: Optional[int] = 192


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adafactor")
    model_max_length: int = field(
        default=8192,
        metadata={"help": "Maximum sequence length"},
    )
    max_steps: int = field(
        default=1000000,
        metadata={"help": "Maximum training steps"},
    )
    save_steps: int = field(
        default=8000,
        metadata={"help": "save ckpt per steps"},
    )
    campplus_onnx_path: str = field(
        default='',
        metadata={"help": "speaker embedding extractor onnx model path"},
    )


@dataclass
class DataArguments:
    data_path: str = field(default="",
                           metadata={"help": "Path to the training data."})
    eval_data_path: str = field(
        default="", metadata={"help": "Path to the evaluation data."})
