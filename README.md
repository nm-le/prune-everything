# prune-everything

1. Install packages through `cd lmms-eval & pip install -e .` For reference, I included `requirements.txt` which is the exact clone of the environment I am running on. Refer to this file if needed (for example, `transformers==4.56.2`)
2. Navigate to `(your environment)/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py` and replace that file with the version in this repo
3. Navigate to `(your environment)/lib/python3.10/site-packages/transformers/integrations/flash_attention.py` and replace that file with the version in this repo
4. Run `bash examples/models/test.sh`
