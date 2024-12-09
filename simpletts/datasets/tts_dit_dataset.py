from functools import partial
from typing import Literal

import s3tokenizer
import torch
from simpletts.datasets.datapipes import WenetRawDatasetSource
from simpletts.datasets.tts_llm_dataset import (decode_wav, extract_spkemb,
                                                filter_by_length)
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

    mels_list = [sample['mel'] for sample in data]
    mels, mels_lens = s3tokenizer.padding(mels_list)

    spks_embs = [sample['spk_emb'] for sample in samples]
    spk_emb = torch.stack(spks_embs, dim=0)

    return {'speech': mels, 'speech_lens': mels_lens, "spk_emb": spk_emb}


def compute_feature(sample, mel_extractor_fn):
    wav, sr = sample['wav'], sample['sample_rate']
    with torch.no_grad():
        assert sr == mel_extractor_fn.sample_rate
        mel = mel_extractor_fn(wav)
    sample['mel'] = mel
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

    # TODO: make it configureable
    mel_extractor = MelSpectrograms(22050,
                                    n_fft=1024,
                                    hop_length=256,
                                    n_mels=80)
    dataset = dataset.map(
        partial(compute_feature, mel_extractor_fn=mel_extractor))
    dataset = dataset.map(partial(extract_spkemb, spk_extractor=spk_extractor))
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
