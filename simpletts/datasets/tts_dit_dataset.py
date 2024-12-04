from functools import partial
from typing import Literal

import s3tokenizer
import torch
import torchaudio
from simpletts.datasets.datapipes import WenetRawDatasetSource
from simpletts.datasets.tts_llm_dataset import (compute_feature, decode_wav,
                                                extract_spkemb,
                                                filter_by_length, resample)
from torch.utils.data import DataLoader


class MelSpectrograms(torch.nn.Module):

    def __init__(
            self,
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=100,
            padding: Literal["center", "same"] = "center",
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.sample_rate = sample_rate
        self.padding = padding
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=padding == "center",
            power=1,
        )

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return super().__call__(audio)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.to(self.device)
        mel: torch.Tensor = self.mel_spec(audio)
        features = torch.log(torch.clip(mel, min=1e-5))
        return features


def padding(data):
    samples = data

    speech_list = [sample['speech'] for sample in data]
    speech, speech_lens = s3tokenizer.padding(speech_list)

    mels_list = [sample['mel'] for sample in data]
    mels_lens = torch.tensor([sample['mel'].size(0) for sample in samples],
                             requires_grad=False)
    mels = torch.nn.utils.rnn.pad_sequence(mels_list,
                                           batch_first=True,
                                           padding_value=0)

    spks_embs = [sample['spk_emb'] for sample in samples]
    spk_emb = torch.stack(spks_embs, dim=0)

    return {
        'speech': speech,
        'speech_lens': speech_lens,
        "spk_emb": spk_emb,
        'mels': mels,
        'mels_lens': mels_lens
    }


def compute_mels(sample, mel_extractor_fn):
    wav, sr = sample['wav'], sample['sample_rate']
    with torch.no_grad():
        assert sr == mel_extractor_fn.sample_rate
        mel = mel_extractor_fn(wav)
    sample['mel'] = mel.squeeze(0).transpose(0, 1)  # [T,n_mels]
    return sample


def init_dataset_and_dataloader(files,
                                batch_size,
                                num_workers,
                                prefetch,
                                shuffle,
                                steps,
                                drop_last=False,
                                spk_extractor=None,
                                seed=2025):

    dataset = WenetRawDatasetSource(files, cycle=steps, shuffle=shuffle)
    # TODO: stage2 shuffle

    dataset = dataset.map(decode_wav)
    dataset = dataset.filter(filter_by_length)

    dataset = dataset.map(partial(
        resample, resample_rate=16000))  # for spk_emb extractor
    dataset = dataset.map(partial(extract_spkemb, spk_extractor=spk_extractor))
    dataset = dataset.map(compute_feature)

    # TODO: make it configureable
    mel_extractor = MelSpectrograms(22050,
                                    n_fft=1024,
                                    hop_length=256,
                                    n_mels=80)
    dataset = dataset.map(partial(resample,
                                  resample_rate=22050))  # for dit mels
    dataset = dataset.map(partial(compute_mels,
                                  mel_extractor_fn=mel_extractor))
    dataset = dataset.batch(batch_size,
                            wrapper_class=padding,
                            drop_last=drop_last)

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            persistent_workers=True,
                            prefetch_factor=prefetch,
                            generator=generator)
    return dataset, dataloader
