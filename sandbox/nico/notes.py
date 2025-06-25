"""
iree-turbine/iree/turbine/ops/insert_slice.py :: l 53 registers custom op

1. custom op works like a regular Python PT op, specified by

    signature = (
        "insert_slice(Tensor src, Tensor dst, int[] offset, int[] strides) -> (Tensor)"
    )

2. generates whatever thanks to `def generate`

3. variant selected by `def select`

This was meant to be the entry point for tkl to connect to the `def generate`.

Could also connect to instrinsics to get ggml-style Q5.

sharktank/models/llm/llm.py is a reimpl of LLM in PT that preconditions for IREE, uses turbine to trace etc

@fxb.export_program like aot.exprt but with multiple entry points

Cross custom-op mechanism is better discussed through sharktank (@1:02:00).
"""