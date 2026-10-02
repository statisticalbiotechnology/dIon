"""DRAFT CODE FOR DATA AUGMENTATION BY PAIR CONCATENATION. NOT TESTED YET."""

import torch


class PairConcatenationAugmentation:
    def __init__(
        self,
        padding_value=0,
    ):
        """
        Usage:
            aug = PairConcatenationAugmentation()

            aug_sequences = aug(sequences, lengths)

        Args:
            padding_value (int): Value to use for padding.
        """
        self.padding_value = padding_value

    def _concatenate_pairs(self, sequences, lengths):
        """
        Concatenate consecutive pairs of spectra to form new combined spectra.

        Args:
            sequences (torch.Tensor): Input sequences of shape (batch_size, seq_len, embed_dim).
            lengths (torch.Tensor): Lengths of sequences in the batch.

        Returns:
            tuple: Concatenated sequences, corresponding padding masks, and pair indices.
        """
        batch_size, seq_len, embed_dim = sequences.shape

        # We will concatenate each spectrum with the next one in the batch
        num_pairs = batch_size - 1  # Concatenate consecutive pairs
        concatenated_sequences = []
        concatenated_lengths = []
        padding_masks = []

        for i in range(num_pairs):
            seq_1, len_1 = sequences[i], lengths[i]
            seq_2, len_2 = sequences[i + 1], lengths[i + 1]

            # Concatenate the two spectra
            combined_len = len_1 + len_2
            combined_seq = torch.cat((seq_1[:len_1], seq_2[:len_2]), dim=0)

            # Create padding for the concatenated spectrum to fit max length in batch
            padded_combined_seq = torch.full(
                (seq_len * 2, embed_dim), self.padding_value, device=sequences.device
            )
            padded_combined_seq[:combined_len] = combined_seq

            # Create padding mask for the concatenated spectrum
            padding_mask = torch.ones(
                seq_len * 2, dtype=torch.bool, device=sequences.device
            )
            padding_mask[:combined_len] = (
                False  # False for real values, True for padding
            )

            concatenated_sequences.append(padded_combined_seq)
            concatenated_lengths.append(combined_len)
            padding_masks.append(padding_mask)

        concatenated_sequences = torch.stack(concatenated_sequences)
        concatenated_lengths = torch.tensor(
            concatenated_lengths, device=sequences.device
        )
        padding_masks = torch.stack(padding_masks)

        return concatenated_sequences, concatenated_lengths, padding_masks

    def __call__(self, sequences, lengths):
        """
        Perform augmentation by concatenating consecutive spectra pairs.

        Args:
            sequences (torch.Tensor): Input sequences of shape (batch_size, seq_len, embed_dim).
            lengths (torch.Tensor): Lengths of sequences in the batch.

        Returns:
            tuple: Augmented sequences, concatenated lengths, and padding masks.
        """
        lengths = lengths.squeeze(-1)
        if any(lengths < 1):
            print("Warning: Found empty spectrum. Can lead to unexpected behaviour.")

        # Concatenate consecutive pairs of spectra
        augmented_sequences, augmented_lengths, padding_masks = self._concatenate_pairs(
            sequences, lengths
        )

        return augmented_sequences, augmented_lengths, padding_masks


if __name__ == "__main__":
    import pytorch_lightning as pl

    pl.seed_everything(0)
    # Example usage
    batch_size, seq_len, embed_dim = 4, 10, 2
    sequences = torch.randn(batch_size, seq_len, embed_dim)
    lengths = torch.tensor([10, 9, 7, 5]).long()

    # Simulate pad tokens (= 0)
    mask = torch.arange(seq_len).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
    sequences[mask.unsqueeze(-1).expand_as(sequences)] = 0

    # Use pair concatenation augmentation
    augmentation = PairConcatenationAugmentation(padding_value=0)

    augmented_sequences, augmented_lengths, padding_masks = augmentation(
        sequences, lengths
    )

    print(f"Original Sequence Shape = {sequences.shape}")
    print(f"Original Sequence Peak Lengths = {lengths}")
    print(f"Augmented Sequence Shape = {augmented_sequences.shape}")
    print(f"Augmented Sequence Peak Lengths = {augmented_lengths}")
    for idx, pad_mask in enumerate(padding_masks):
        print(
            f"Augmented Sequence {idx+1} (~pad_mask).sum(-1) = length of real peaks = {(~pad_mask).sum(-1)}"
        )
