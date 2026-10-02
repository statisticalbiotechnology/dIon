import warnings
from src.models.dc_models.depthcharge_peptides import PeptideTransformerDecoder as DcPeptideTransformerDecoder
from src.models.dc_models.depthcharge_peptides import generate_tgt_mask
from src.models.sinusoidal import PositionalEncoder
import torch
import torch.nn as nn


def generate_causal_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square causal mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)


class PeptideTransformerDecoder(DcPeptideTransformerDecoder):
    def __init__(
        self,
        tokenizer,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
        use_mass: bool = True,
        use_charge: bool = True,
        max_seq_len: int = 31,
        cross_attend: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.amod_dict = tokenizer.index
        self.input_dict = tokenizer.index
        self.SOS = tokenizer.bos_token_id
        self.output_dict = tokenizer.index
        self.NT = tokenizer.pad_token_id
        self.EOS = tokenizer.eos_token_id
        self.output_dict_rev = {b: a for a, b in self.output_dict.items()}
        self.output_dict_rev[self.EOS] = "$"

        self.num_input_tokens = tokenizer.vocab_size
        self.num_output_tokens = tokenizer.vocab_size

        super().__init__(
            self.num_input_tokens
            - 1,  # decoder base adds one for the start token
            d_model,
            nhead,
            dim_feedforward,
            n_layers,
            dropout,
            positional_encoder,
            max_charge,
        )

        if not cross_attend:
            del self.transformer_decoder
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
            )

            self.transformer_decoder = torch.nn.TransformerEncoder(
                layer, num_layers=n_layers
            )

        self.d_model = d_model
        self.use_mass = use_mass
        self.use_charge = use_charge
        self.max_seq_len = max_seq_len
        self.cross_attend = cross_attend

        # Override the final projection with
        # the correct num_classes
        self.final = torch.nn.Linear(
            d_model,
            self.num_output_tokens,
        )

    def detokenize(self, integer_sequence):
        return [self.output_dict_rev[integer] for integer in integer_sequence.tolist()]

    def forward(
        self,
        tokens: torch.Tensor | None,
        precursors: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
        peptide_lengths: torch.Tensor | None = None,
        causal: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences."""
        if tokens is None:
            tokens = torch.tensor([[]]).to(self.device)

        batch_size = memory.shape[0]
        token_ids = tokens

        tokens = self.aa_encoder(tokens)

        mass = precursors[:, 0:1] if self.use_mass else None
        charge = precursors[:, 1:2] if self.use_charge else None

        if mass is not None:
            masses = self.mass_encoder(mass)
            if not self.use_mass:
                warnings.warn("Not trained to include precursor mass")
        else:
            masses = 0
        if charge is not None:
            charges = self.charge_encoder(charge.int() - 1)
            charges = charges[:, None, :]
            if not self.use_charge:
                warnings.warn("Not trained to include precursor charge")
        else:
            charges = 0

        if mass is not None or charge is not None:
            precursors = masses + charges
        else:
            empty = [[[]] * self.d_model] * batch_size
            precursors = torch.tensor(empty, device=self.device).transpose(-2, -1)

        num_precursor_tokens = precursors.shape[1]
        tgt = torch.cat([precursors, tokens], dim=1)
        tgt = self.positional_encoder(tgt)

        cls_token = kwargs.get("cls_token")
        if cls_token is not None:
            tgt = torch.cat([cls_token, tgt], dim=1)
            num_precursor_tokens += cls_token.shape[1]

        tgt_mask = (
            generate_causal_tgt_mask(tgt.shape[1]).to(self.device) if causal else None
        )

        if self.cross_attend:
            dec_embeds = self.transformer_decoder.forward(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            assert (
                cls_token is not None
            ), "Can only forward without cross-attention if cls_token is given"
            dec_embeds = self.transformer_decoder.forward(
                src=tgt,
                mask=tgt_mask,
                src_key_padding_mask=None,
            )
        logits = self.final(dec_embeds)
        logits = logits[:, num_precursor_tokens:, :]

        if peptide_lengths is None:
            _ = None
        else:
            _ = self.get_padding_mask(
                token_ids,
                peptide_lengths,
                0,
            )
        return logits, token_ids

    def get_padding_mask(self, input_intseq, peptide_lengths, num_extra_tokens):
        batch_size = input_intseq.shape[0]

        total_len = peptide_lengths + num_extra_tokens
        inds = (
            torch.arange(
                input_intseq.shape[1] + num_extra_tokens, device=input_intseq.device
            )
            .unsqueeze(0)
            .repeat((batch_size, 1))
        )
        pad_mask = inds > total_len
        return pad_mask

    def initialize_sequence(self, batch_size):
        return torch.tensor([[self.SOS]], dtype=torch.long, device=self.device).repeat(
            (batch_size, 1)
        )

    def predict_sequence(
        self,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor | None,
        precursors: torch.Tensor,
        causal: bool = True,
        **kwargs,
    ):
        batch_size = memory.shape[0]

        # Initialize output tensors
        input_intseq = torch.tensor(
            [[self.SOS]], dtype=torch.long, device=self.device
        ).repeat((batch_size, 1))

        logits = torch.zeros(
            batch_size, self.max_seq_len, self.num_output_tokens
        ).type_as(memory)

        # Gather predictions (fixed length loop)
        for i in range(0, self.max_seq_len):
            cur_logits, _ = self.forward(
                input_intseq,
                precursors,
                memory,
                memory_key_padding_mask,
                causal=causal,
                **kwargs,
            )

            predictions = self.greedy(cur_logits[:, i])
            logits[:, i, :] = cur_logits[:, i]

            input_intseq = torch.cat([input_intseq, predictions], dim=1)

        return input_intseq[:, 1:], logits

    def greedy(self, predict_logits: torch.Tensor):
        return predict_logits.argmax(dim=-1, keepdim=True).type(torch.int32)


def dc_decoder_tiny(
    amod_dict, d_model=256, dropout=0, cross_attend=True, max_charge=10, **kwargs
):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=2,
        dim_feedforward=128,
        n_layers=1,
        positional_encoder=True,
        max_charge=max_charge,
        use_mass=True,
        use_charge=True,
        dropout=dropout,
        cross_attend=cross_attend,
    )
    return model


def dc_decoder_base(
    amod_dict,
    d_model=256,
    dropout=0,
    cross_attend=True,
    max_seq_len=31,
    max_charge=10,
    **kwargs
):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=512,
        n_layers=9,
        positional_encoder=True,
        max_charge=max_charge,
        use_mass=True,
        use_charge=True,
        dropout=dropout,
        cross_attend=cross_attend,
        max_seq_len=max_seq_len,
    )
    return model


def dc_decoder_deeper(
    amod_dict, d_model=256, dropout=0.25, cross_attend=True, max_charge=10, **kwargs
):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=512,
        n_layers=15,
        dropout=dropout,
        positional_encoder=True,
        max_charge=max_charge,
        use_mass=True,
        use_charge=True,
        cross_attend=cross_attend,
    )
    return model


def dc_decoder_jl(
    amod_dict,
    d_model=256,
    dropout=0.1,
    cross_attend=True,
    max_seq_len=100,
    max_charge=10,
    **kwargs
):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=d_model,
        n_layers=9,
        dropout=dropout,
        positional_encoder=True,
        max_charge=max_charge,
        use_mass=True,
        use_charge=True,
        cross_attend=cross_attend,
        max_seq_len=max_seq_len,
    )
    return model


def dc_casanovo_decoder(
    amod_dict,
    d_model=512,
    dropout=0,
    cross_attend=True,
    max_seq_len=100,
    max_charge=10,
    **kwargs
):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=1024,
        n_layers=9,
        positional_encoder=True,
        max_charge=max_charge,
        use_mass=True,
        use_charge=True,
        dropout=dropout,
        cross_attend=cross_attend,
        max_seq_len=max_seq_len,
    )
    return model
