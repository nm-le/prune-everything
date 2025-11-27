# Run and exactly reproduce qwen2vl results!
# mme as an example
export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#     --model qwen2_vl \
#     --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=True \
#     --tasks mmmu_pro \
#     --batch_size 1

# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#   --model qwen2_5_vl --force_simple \
#   --model_args=pretrained=Osilly/Vision-R1-7B,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False,kvpress_method=streaming,kvpress_ratio=0.5 \
#   --tasks mme \
#   --batch_size 1 \
#   --limit 10 \
#   --log_samples \
#   --log_samples_suffix reproduce \
#   --output_path ./logs/qwen25vl/mmstar_test/


# accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
#   --model qwen2_5_vl --force_simple \
#   --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
#   --tasks mme \
#   --batch_size 1 \
#   --limit 10 \
#   --log_samples \
#   --log_samples_suffix reproduce \
#   --output_path ./logs/qwen25vl/mmstar_test/


QWEN=Qwen/Qwen2.5-VL-7B-Instruct
VISIONR1=Osilly/Vision-R1-7B
PYTHONPATH=/home/minhle/projects/aip-btaati/minhle:$PYTHONPATH \


accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
  --model qwen2_5_vl --force_simple \
  --model_args "\
pretrained=$QWEN,\
max_pixels=12845056,\
attn_implementation=sdpa,\
interleave_visuals=False,"\
  --tasks mmstar \
  --batch_size 1 \
  --limit 100 \
  --log_samples \
  --log_samples_suffix reproduce \
  --output_path ./logs/qwen25vl/sep29/mmstar_100/