import json
from functools import partial

import s3tokenizer
import torch
import torchaudio
from simpletts.datasets.datapipes import WenetRawDatasetSource
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def decode_wav(sample):
    obj = json.loads(sample['line'])
    filepath, txt = obj['wav'], obj['txt']
    audio, sample_rate = torchaudio.load(filepath)
    return {
        'wav': audio,
        "sample_rate": sample_rate,
        'txt': txt,
    }


def resample(sample, resample_rate=16000):
    """ Resample sample.
        Inplace operation.

        Args:
            sample: {key, wav, label, sample_rate}
            resample_rate: target resample rate

        Returns:
            {key, wav, label, sample_rate}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    if sample_rate != resample_rate:
        sample['sample_rate'] = resample_rate
        sample['wav'] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate)(waveform)
    return sample


def compute_feature(sample):
    wav = sample['wav'][0]  # get first channel
    mel = s3tokenizer.log_mel_spectrogram(wav)
    sample['speech'] = mel
    return sample


def filter_by_length(sample, max_seconds=30):
    wav = sample['wav']
    sr = sample['sample_rate']
    if wav.shape[0] / sr <= max_seconds:
        return True
    return False


def extract_spkemb(sample, spk_extractor=None):
    if spk_extractor is None:
        sample['spk_emb'] = None
        return sample
    sample['spk_emb'] = spk_extractor(sample['wav'], sample['sample_rate'])
    return sample


def padding(data, pad_value=0):
    samples = data

    mels_list = [sample['speech'] for sample in data]
    mels, mels_lens = s3tokenizer.padding(mels_list)

    labels = [
        torch.tensor(sample['text'], dtype=torch.int64) for sample in samples
    ]
    labels = pad_sequence(labels, batch_first=True, padding_value=pad_value)
    label_lengths = torch.tensor([x.size(0) for x in labels],
                                 dtype=torch.int32)

    spks_embs = [sample['spk_emb'] for sample in samples]
    spk_emb = torch.stack(spks_embs, dim=0)
    return {
        'speech': mels,
        'speech_lens': mels_lens,
        "text": labels,
        "text_lens": label_lengths,
        "spk_emb": spk_emb
    }


def tokenizeOp(sample, tokenizer, add_bos=True, add_eos=True):
    """pretrain
    """
    obj = sample
    text = obj['txt']
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if add_bos:
        bos_id = tokenizer.bos_token_id
        bos_token = tokenizer.convert_ids_to_tokens([bos_id])[0]
        tokens = [bos_token] + tokens
        ids = [bos_id] + ids
    if add_eos:
        eos_id = tokenizer.eos_token_id
        eos_token = tokenizer.convert_ids_to_tokens([eos_id])[0]
        tokens = tokens + [eos_token]
        ids = ids + [eos_id]
    sample['text'] = ids
    sample['text_lens'] = len(ids)
    return sample


def init_dataset_and_dataloader(files,
                                tokenizer,
                                batch_size,
                                num_workers,
                                prefetch,
                                shuffle,
                                steps,
                                drop_last=False,
                                spk_extractor=None,
                                sample_rate=16000,
                                seed=2025):

    dataset = WenetRawDatasetSource(files, cycle=steps, shuffle=shuffle)
    # TODO: stage2 shuffle

    dataset = dataset.map(decode_wav)
    dataset = dataset.filter(filter_by_length)
    dataset = dataset.map(partial(resample, resample_rate=sample_rate))
    dataset = dataset.map(compute_feature)
    dataset = dataset.map(partial(tokenizeOp, tokenizer=tokenizer))
    dataset = dataset.map(partial(extract_spkemb, spk_extractor=spk_extractor))
    dataset = dataset.batch(batch_size,
                            wrapper_class=partial(
                                padding, pad_value=tokenizer.pad_token_id),
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
