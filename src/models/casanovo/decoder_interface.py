from .transformers import (
    PositionalEncoder,
    FloatEncoder,
)
from . import utils
import torch


def generate_tgt_mask(sz):
    """Generate a square mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return ~torch.triu(torch.ones(sz, sz, dtype=torch.bool)).transpose(0, 1)


class _PeptideTransformer(torch.nn.Module):
    """A transformer base class for peptide sequences.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality to represent the amino acids in a peptide
        sequence.
    pos_encoder : bool
        Use positional encodings for the amino acid sequence.
    vocab_size : int
        The number of amino acids (size of the vocabulary).
    max_charge : int
        The maximum charge to embed.
    padding_idx : int
        The index used for padding tokens.
    """

    def __init__(
        self,
        dim_model,
        pos_encoder,
        vocab_size,
        max_charge,
        padding_idx=0,
    ):
        super().__init__()
        self.reverse = False
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx

        if pos_encoder:
            self.pos_encoder = PositionalEncoder(dim_model)
        else:
            self.pos_encoder = torch.nn.Identity()

        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        # vocab_size already includes special tokens (PAD/BOS/EOS/UNK)
        self.aa_encoder = torch.nn.Embedding(
            vocab_size,
            dim_model,
            padding_idx=padding_idx,
        )


class PeptideDecoder(_PeptideTransformer):
    def __init__(
        self,
        dim_model=128,
        n_head=8,
        dim_feedforward=1024,
        n_layers=1,
        dropout=0,
        pos_encoder=True,
        reverse=False,
        vocab_size=None,
        max_charge=5,
        padding_idx=0,
    ):
        """Initialize a PeptideDecoder"""
        super().__init__(
            dim_model=dim_model,
            pos_encoder=pos_encoder,
            vocab_size=vocab_size,
            max_charge=max_charge,
            padding_idx=padding_idx,
        )
        self.reverse = reverse
        self.padding_idx = padding_idx

        # Additional model components
        self.mass_encoder = FloatEncoder(dim_model)
        layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_decoder = torch.nn.TransformerDecoder(
            layer,
            num_layers=n_layers,
        )

        self.final = torch.nn.Linear(dim_model, vocab_size)

    def forward(self, tokens, precursors, memory, memory_key_padding_mask, **kwargs):
        """
        Forward pass of the decoder.

        Parameters
        ----------
        tokens : torch.Tensor of shape (batch_size, seq_len)
            Integer token sequences representing amino acids.
        precursors : torch.Tensor of shape (batch_size, 3)
            The measured precursor mass (axis 0), charge (axis 1), and m/z (axis 2) of each spectrum.
        memory : torch.Tensor of shape (batch_size, n_peaks, dim_model)
            The encoder outputs.
        memory_key_padding_mask : torch.Tensor of shape (batch_size, n_peaks)
            The mask that indicates which elements of `memory` are padding.

        Returns
        -------
        torch.Tensor
            The raw output logits for each amino acid prediction.
        torch.Tensor
            The input tokens (for potential use in loss calculation or further processing).
        """
        # Prepare mass and charge
        masses = self.mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]

        # Feed through model:
        if tokens is None or tokens.size(1) == 0:
            tgt = precursors
            tgt_key_padding_mask = torch.zeros(
                (tgt.shape[0], 1), dtype=torch.bool, device=memory.device
            )
        else:
            tgt = torch.cat([precursors, self.aa_encoder(tokens)], dim=1)
            precursors_mask = torch.zeros(
                (tokens.shape[0], 1), dtype=torch.bool, device=memory.device
            )
            tokens_padding_mask = tokens == self.padding_idx
            tgt_key_padding_mask = torch.cat(
                [precursors_mask, tokens_padding_mask], dim=1
            )

        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(memory.device)

        preds = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=True,
        )
        return self.final(preds), tokens


def casanovo_decoder(
    tokenizer,
    d_model=512,
    dropout=0,
    cross_attend=True,
    max_seq_len=100,
    max_charge=10,
    **kwargs
):
    if not cross_attend:
        raise NotImplementedError("This decoder only supports cross_attend=True. ")

    padding_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    model = PeptideDecoder(
        dim_model=d_model,
        n_head=8,
        dim_feedforward=1024,
        n_layers=9,
        dropout=dropout,
        pos_encoder=True,
        reverse=False,
        vocab_size=vocab_size,
        max_charge=max_charge,
        padding_idx=padding_idx,
    )
    return model


def casanovo_decoder_tiny(
    tokenizer,
    d_model=512,
    dropout=0,
    cross_attend=True,
    max_seq_len=100,
    max_charge=10,
    **kwargs
):
    if not cross_attend:
        raise NotImplementedError("This decoder only supports cross_attend=True. ")

    padding_idx = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size
    model = PeptideDecoder(
        dim_model=d_model,
        n_head=2,
        dim_feedforward=128,
        n_layers=2,
        dropout=dropout,
        pos_encoder=True,
        reverse=False,
        vocab_size=vocab_size,
        max_charge=max_charge,
        padding_idx=padding_idx,
    )
    return model
