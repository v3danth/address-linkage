"""Link people by where they live: same address, same house, same building or complex, neighbours.

Uses the Amazon ML Challenge entity-resolution pipeline (blocking, dense retrieval, staged LightGBM)
on home addresses only; names and family details are used only to report and evaluate results.
"""
