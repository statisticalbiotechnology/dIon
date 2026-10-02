# KitchenSink V3 Reference

This directory preserves the exact KitchenSink v3 construction scripts used
for the current corpus. It is reference material for the dIon-de-novo-labeled-v1 (DNLv1)
rebuild; it is not a new default materialization entry point.

The files are kept byte-for-byte identical to the originals:

| File | SHA-256 |
| --- | --- |
| `build_kitchensink_v3.py` | `1ddda109402bcfbce97b527d43e9fb60c690e6918fce1e66aeefe4a2b04d9552` |
| `cleanup_kitchensink_v3_lance.py` | `88294623309322394352f7fe6f9915c34a0dae4b64e45d51f4053d9b94bae9df` |
| `config.public.json` | `c78ad6fbaa91da26df19fef1ae8d5476a81031f4a9b7c58dce89d482fcd9b890` |

The v3 builder performs staged source selection, split resolution,
materialization, and verification. The cleanup script applies final
peptidoform split precedence (`test`, then `val`, then `train`) and exact
m/z-peak-list deduplication. A v5 builder should reuse these policies
explicitly while replacing the legacy `mskb_v5_seed` terminology and source
membership with the revised `mskb_final` construction.
