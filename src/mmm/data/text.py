"""
mmm/data/text.py
"""

import torch

from torch.utils.data import Dataset, IterableDataset


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_length: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.tokens = torch.randint(
            self.vocab_size,
            size=(len(self), self.seq_length + 1),
            # Set a seed to make this toy dataset the same on each rank
            # Fabric will add a `DistributedSampler` to shard the data
            # correctly
            generator=torch.Generator().manual_seed(42),
        )

    def __len__(self) -> int:
        return 128

    def __getitem__(self, item: int):
        return self.tokens[item]


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.input_characters = seq_length * chars_per_token * num_of_sequences

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.input_characters:
                    m = f'Buffer full: {buffer_len}>={self.input_characters:.0f}'
                    print(m)
                    break
                try:
                    m = f'Fill buffer: {buffer_len}<{self.input_characters:.0f}'
                    print(m)
                    buffer.append(next(iterator)['text'])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    iterator = iter(self.dataset)

            all_token_ids = []
            tokenized_inputs = self.tokenizer(buffer, truncation=False)
            for tokenized_input in tokenized_inputs['input_ids']:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])

            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    yield torch.tensor(input_ids)
