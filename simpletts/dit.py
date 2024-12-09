from typing import Dict

import s3tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel


class TimestepEmbedder(nn.Module):
    """
    Rotational positional encoding for timestep.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int):
        super(TimestepEmbedder, self).__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.network = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.network(t_freq)
        return t_emb

    @staticmethod
    def timestep_embedding(t: torch.Tensor,
                           freq_emb_size: int,
                           max_period=10000) -> torch.Tensor:
        if not torch.is_floating_point(t):
            t = t.float()

        # TODO: replace with ROPE
        half = freq_emb_size // 2
        freqs = torch.exp(-torch.log(torch.tensor(float(max_period))) *
                          torch.arange(0, half, dtype=t.dtype) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)

        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if freq_emb_size % 2 == 1:
            extra_zeros = torch.zeros_like(embedding[:, :1])
            embedding = torch.cat([embedding, extra_zeros], dim=-1)

        return embedding


class SimpleDIT(PreTrainedModel):
    supports_gradient_checkpointing = True
    """ Only In-context conditioning is implemented
        In-context condition, see https://arxiv.org/pdf/2212.09748 section 3.2
    """
    supports_gradient_checkpointing = True

    def __init__(self, config, llm: nn.Module) -> None:
        super().__init__(config)
        self.llm = llm
        self.config = config

        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)
        self.speech_tokenizer = s3tokenizer.load_model(
            "speech_tokenizer_v1_25hz")
        # self.x_embedder = nn.Linear(1, self.config.hidden_size, bias=True)
        # self.t_embedder = TimestepEmbedder(min(self.dim, 1024), 256)
        # self.y_embedder = LabelEmbedder(self.n_classes, min(self.dim, 1024),
        #                                 self.class_dropout_prob)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        spk_emb: torch.Tensor,
        speech: torch.Tensor,
        speech_lens: torch.Tensor,
        **kwargs,
    ):
        speech_codes, speech_codes_lens = self.speech_tokenizer.quantize(
            speech, speech_lens)
        speech_codes = speech_codes.clone()

        # TODO:
        # spk emb + token + noise
        pass

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()
