import s3tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

from simpletts.utils.common import (combine_shif_left, make_non_pad_mask,
                                    make_pad_mask)


class EmbedMLP(nn.Module):

    def __init__(self, input_size, intermediate_size, output_size):
        super().__init__()
        self.hidden_size = input_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size,
                                   self.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(self.hidden_size,
                                 self.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(self.intermediate_size,
                                   output_size,
                                   bias=False)
        self.act_fn = torch.nn.SiLU(output_size)

    def forward(self, hidden_state):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_state)) *
            self.up_proj(hidden_state))


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


def label_keep_mask(input_lens: torch.Tensor,
                    drop_probs: float = 0.5) -> torch.Tensor:
    """ drop N labels from the input
    """
    import random
    B = input_lens.size(0)
    max_len = input_lens.max()
    keep_ids = torch.zeros(B, max_len, dtype=torch.bool)

    for i, j in enumerate(input_lens):
        if random.random() < drop_probs:
            continue
        index = random.randint(0, int(0.3 * j))
        keep_ids[i, :index] = True

    return keep_ids


class SimpleDIT(PreTrainedModel):
    """
        see https://arxiv.org/pdf/2212.09748 section 3.2
    """
    supports_gradient_checkpointing = True

    def __init__(self, config, llm: nn.Module) -> None:
        super().__init__(config)
        self.llm = llm
        self.config = config

        self.speech_tokenizer = s3tokenizer.load_model(
            "speech_tokenizer_v1_25hz")
        self._keys_to_ignore_on_save = set()
        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)

        self.t_embedder = TimestepEmbedder(config.hidden_size,
                                           config.frequency_embedding_size)
        emb_dim = config.spk_embed_dim + config.hidden_size + config.n_mels
        self.embed_mlp = EmbedMLP(emb_dim, emb_dim * 2, config.hidden_size)
        self.to_mels = torch.nn.Linear(config.hidden_size, config.n_mels)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        spk_emb: torch.Tensor,
        speech: torch.Tensor,
        speech_lens: torch.Tensor,
        mels: torch.Tensor,
        mels_lens: torch.Tensor,
        **kwargs,
    ):
        # 1 speech token embed
        speech_codes, _ = self.speech_tokenizer.quantize(speech, speech_lens)
        speech_codes = speech_codes.clone()
        speech_emb = self.llm.get_input_embeddings()(speech_codes)
        # 1.1 align speech token, mels
        mels_mask_pad = make_non_pad_mask(mels_lens)
        # TODO: fix bug here
        token_emb = F.interpolate(speech_emb.transpose(1, 2),
                                  size=mels_lens.max(),
                                  mode='linear')
        token_emb = token_emb.transpose(1, 2)
        token_emb = token_emb * mels_mask_pad.unsqueeze(2)  # [B,T,D]

        # 2 spk emb
        spk_emb = spk_emb.unsqueeze(1).repeat(
            1,
            mels.shape[1],
            1,
        )

        # 3 [spk_emb,
        #   token_emb]
        conds = torch.cat((spk_emb, token_emb), dim=2)
        # TODO: soundstorm mask strategy here
        keep = label_keep_mask(mels_lens)
        conds = conds * keep.unsqueeze(2)

        # 4 input
        # https://huggingface.co/blog/Isamu136/insta-rectified-flow
        batch_size = speech.size(0)
        t = torch.rand([batch_size], device=speech.device, dtype=speech.dtype)
        t_embed = self.t_embedder(t)
        t = t[:, None, None]
        noise = torch.randn_like(mels)
        zt = (1 - t) * noise + t * mels
        u = mels - noise

        inputs = torch.cat((conds, zt), dim=2)
        inputs = self.embed_mlp(inputs)
        inputs = torch.cat((t_embed.unsqueeze(1), inputs), dim=1)

        # 5 forward llm
        # TODO: dynamic chunk mask
        attention_mask = None
        preds = self.llm(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True)['hidden_states'][-1][:,
                                                            1:, :]  # [B,T,D]
        preds = self.to_mels(preds)
        output_mask = make_non_pad_mask(mels_lens).unsqueeze(2)

        # 6 compute_loss
        loss = F.mse_loss(
            preds * output_mask, u * output_mask,
            reduction="sum") / (torch.sum(output_mask) * u.shape[1])
        return {"loss": loss}

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()
