python tests/kernel/matmul_test.py
python tests/kernel/fused_attention_test.py

PATH=$(dirname $(which python)):${PATH} lit lit_tests/kernel/tkl -v