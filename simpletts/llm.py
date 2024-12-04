from typing import Dict

import s3tokenizer
import torch
from torch import nn
from transformers import PreTrainedModel

from simpletts.utils.common import (add_sos_eos, combine_shif_left,
                                    make_non_pad_mask, make_pad_mask)


class TTSLLM(PreTrainedModel):

    supports_gradient_checkpointing = True

    def __init__(self, config, llm: nn.Module, special_tokens: Dict) -> None:
        super().__init__(config)
        self.llm = llm
        self.speech_tokenizer = s3tokenizer.load_model(
            "speech_tokenizer_v1_25hz")
        self.speech_tokenizer.freeze()

        self.config = config

        self.spk_embed_affine_layer = torch.nn.Linear(
            self.config.spk_embed_dim, self.config.hidden_size)
        self.sos_audio = special_tokens['<|startofaudio|>']
        self.eos_audio = special_tokens['<|endofaudio|>']
        self.pad = special_tokens['pad']

        self._keys_to_ignore_on_save = set()
        for k in self.speech_tokenizer.state_dict().keys():
            self._keys_to_ignore_on_save.add('speech_tokenizer.' + k)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(
        self,
        spk_emb: torch.Tensor,
        speech: torch.Tensor,
        speech_lens: torch.Tensor,
        text: torch.Tensor,
        text_lens: torch.Tensor,
        **kwargs,
    ):
        # 1 speech tokens
        speech_codes, speech_codes_lens = self.speech_tokenizer.quantize(
            speech, speech_lens)
        speech_codes = speech_codes.clone()
        # offset, we assume vocab layout:
        # text token, sos_audio_tokens, audio tokens, eos_audio_tokens
        speech_codes = speech_codes + self.sos_audio
        speech_token_in, speech_token_out = add_sos_eos(
            speech_codes,
            self.sos_audio,
            self.eos_audio,
            -100,
        )
        speech_codes_lens = speech_codes_lens + 2

        # 2  speaker embed
        spk_emb = self.spk_embed_affine_layer(spk_emb)

        # 3 text tokens + speech tokens
        # NOTE:
        #   text: <bos>....<eos>
        # [text_emb, speech_codes_emb]
        text_padding_mask = make_pad_mask(text_lens)
        speech_in_padding_mask = make_pad_mask(speech_codes_lens - 1)
        text_speech, _ = combine_shif_left(text.unsqueeze(2),
                                           speech_token_in.unsqueeze(2),
                                           text_padding_mask,
                                           speech_in_padding_mask)
        text_speech = text_speech.squeeze(2)
        # [text_emb, <startof_audio> ....]
        text_speech_emb = self.llm.get_input_embeddings()(text_speech)

        # 4 speaker embed + text tokens + speech tokens
        # [spk_emb, text_emb, speech_codes_emb]
        #  spke_emb <bos>...<eos> <|start_of_audio|> speech tokens ...
        input_embeds = torch.cat((spk_emb.unsqueeze(1), text_speech_emb),
                                 dim=1)
        input_lens = text_lens + (speech_codes_lens - 1) + 1

        # 5 targets
        ignore_id_prepand = torch.zeros(input_embeds.size(0),
                                        1 + text.size(1),
                                        dtype=torch.long,
                                        device=spk_emb.device) + (-100)
        ignore_input_valid_padding_mask = make_pad_mask(1 + text_lens)
        targets, _ = combine_shif_left(
            ignore_id_prepand.unsqueeze(2),
            speech_token_out.unsqueeze(2),
            ignore_input_valid_padding_mask,
            speech_in_padding_mask,
        )
        targets = targets.squeeze(2)
        assert targets.shape[1] == input_embeds.shape[1]

        # 6 attention mask
        attention_mask = make_non_pad_mask(input_lens)
        # 7 forward and loss
        return self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=targets,
        )

    def enable_input_require_grads(self):
        self.llm.enable_input_require_grads()
