export HF_HOME="~/.cache/huggingface"

QWEN="Qwen/Qwen2.5-VL-7B-Instruct"
VISIONR1="Osilly/Vision-R1-7B"
VLRETHINKER="TIGER-Lab/VL-Rethinker-7B"
OPENVL="ydeng9/OpenVLThinker-7B-v1.2"
VISIONR1_32="Osilly/Vision-R1-32B"
VISIONR1_CI="Osilly/Vision-R1-CI-7B"
VICRIT="zyang39/ViCrit-7B"

date_tag=$(date +"%b%d" | tr '[:upper:]' '[:lower:]')

# run mode: 'seq' (default) or 'con'
mode="${1:-seq}"
if [[ "$mode" != "seq" && "$mode" != "con" ]]; then
  echo "Usage: $0 [seq|con]" >&2
  echo "  seq  run experiments sequentially (default)" >&2
  echo "  con  run experiments concurrently" >&2
  exit 1
fi

# limit is no longer required
required_vars=(ckpt task)

unset_all() {
  # Only unset the things we manage here
  for v in checkpoint ckpt task limit \
           enable_visionzip visionzip_ratio \
           enable_kdvz kdvz_ratio \
           enable_kd_prefill prefill_anchor prefill_ratio prefill_prune_after_layer \
           enable_kd_decode decode_anchor decode_ratio decode_prune_window decode_prune_after_layer \
           majority_vote temperature output_path device; do
    unset "$v" || true
  done
}

set_defaults_optional() {
  majority_vote="${majority_vote:-1}"
  temperature="${temperature:-0.0}"

  enable_visionzip="${enable_visionzip:-False}"
  enable_kdvz="${enable_kdvz:-False}"
  enable_keydiff="${enable_keydiff:-False}"
  # NOTE: no default for limit; leave it unset unless provided
}

validate_required() {
  local missing=()
  for v in "${required_vars[@]}"; do
    if [[ -z "${!v-}" ]]; then
      missing+=("$v")
    fi
  done
  if ((${#missing[@]})); then
    echo "Missing required vars: ${missing[*]}" >&2
    exit 1
  fi
}

run_one() {
  set_defaults_optional
  checkpoint="${!ckpt}"
  model_dir="${ckpt,,}"
  validate_required

  config_tags=()

  if [[ "${enable_visionzip}" == "True" ]]; then
    # assume visionzip_ratio is provided when enabled
    config_tags+=("vz${visionzip_ratio//./}")
  fi

  if [[ "${enable_kdvz}" == "True" ]]; then
    # assume kdvz_ratio is provided when enabled
    config_tags+=("kdvz${kdvz_ratio//./}")
  fi

  if [[ "${enable_kd_prefill}" == "True" ]]; then
    # assume {prefill_anchor,prefill_ratio,prefill_prune_after_layer} provided when enabled
    config_tags+=("kpf${prefill_anchor}_${prefill_ratio//./}_i${prefill_prune_after_layer}")
  fi

  if [[ "${enable_kd_decode}" == "True" ]]; then
    # assume {decode_anchor,decode_ratio,decode_prune_window} provided when enabled
    config_tags+=("kdc${decode_anchor}_${decode_ratio//./}_w${decode_prune_window}_i${decode_prune_after_layer}")
  fi

  if [ ${#config_tags[@]} -eq 0 ]; then
    config_tags=("base")
  fi

  if [[ "$majority_vote" -gt 1 ]]; then
    config_tags+=("major${majority_vote}")
  fi

  if [[ "$temperature" != "0.0" ]]; then
    config_tags+=("temp${temperature//./}")
  fi

  config_suffix=$(IFS=_; echo "${config_tags[*]}")

  # Only append limit to the task name if it was provided
  task_with_limit="$task"
  if [[ -n "${limit-}" ]]; then
    task_with_limit="${task}${limit}"
  fi

  output_path="./logs/${date_tag}/${model_dir}/${task_with_limit}_${config_suffix}/"
  # output_path="./logs/oct19/${model_dir}/${task_with_limit}_${config_suffix}/"

  echo "Saving results to: $output_path"
  echo "Using CUDA device: $device"
  sleep 1

  model_args="pretrained=$checkpoint,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False,benchmark=$task,majority_vote=$majority_vote,temperature=$temperature,output_path=$output_path,"

  if [[ "${enable_visionzip}" == "True" ]]; then
    model_args+="enable_visionzip=True,visionzip_ratio=$visionzip_ratio,"
  else
    model_args+="enable_visionzip=False,"
  fi

  if [[ "${enable_kdvz}" == "True" ]]; then
    model_args+="enable_kdvz=True,kdvz_ratio=$kdvz_ratio,"
  else
    model_args+="enable_kdvz=False,"
  fi

  if [[ "${enable_kd_prefill}" == "True" ]]; then
    model_args+="enable_kd_prefill=True,prefill_anchor=$prefill_anchor,prefill_ratio=$prefill_ratio,prefill_prune_after_layer=$prefill_prune_after_layer,"
  else
    model_args+="enable_kd_prefill=False,"
  fi

  if [[ "${enable_kd_decode}" == "True" ]]; then
    model_args+="enable_kd_decode=True,decode_anchor=$decode_anchor,decode_ratio=$decode_ratio,decode_prune_window=$decode_prune_window,decode_prune_after_layer=$decode_prune_after_layer,"
  else
    model_args+="enable_kd_decode=False,"
  fi

  local port=$((12345 + device))


  # Build the base command
  # cmd=(accelerate launch --num_processes=1 --main_process_port="$port" -m lmms_eval
  cmd=(python -m lmms_eval
       --model vision_r1 --force_simple
       --model_args "$model_args"
       --tasks "$task"
       --batch_size 1
       --log_samples
       --log_samples_suffix reproduce
       --output_path "$output_path")

  # Append --limit only if provided
  if [[ -n "${limit-}" ]]; then
    cmd+=(--limit "$limit")
  fi

  CUDA_VISIBLE_DEVICES="$device" "${cmd[@]}"
}

experiments=(

  "device=0 ckpt=OPENVL task=mme limit=1000" \
#   "device=0 ckpt=QWEN task=mme limit=1000 majority_vote=5 temperature=1.0" \

  "device=0 ckpt=OPENVL task=mme limit=1000 \
  enable_kdvz=True kdvz_ratio=0.5 \
  enable_kd_prefill=True prefill_anchor=all prefill_ratio=0.5 prefill_prune_after_layer=8" \
#   "device=1 ckpt=QWEN task=mme limit=1000 majority_vote=5 temperature=1.0 \
#   enable_kdvz=True kdvz_ratio=0.5 \
#   enable_kd_prefill=True prefill_anchor=all prefill_ratio=0.5 prefill_prune_after_layer=8" \

  "device=0 ckpt=OPENVL task=mme limit=1000 \
  enable_kdvz=True kdvz_ratio=0.75 \
  enable_kd_prefill=True prefill_anchor=all prefill_ratio=0.5 prefill_prune_after_layer=8" \
#   "device=2 ckpt=QWEN task=mme limit=1000 majority_vote=5 temperature=1.0 \
#   enable_kdvz=True kdvz_ratio=0.75 \
#   enable_kd_prefill=True prefill_anchor=all prefill_ratio=0.5 prefill_prune_after_layer=8" \

  "device=0 ckpt=OPENVL task=mme limit=1000 \
  enable_kdvz=True kdvz_ratio=0.75 \
  enable_kd_prefill=True prefill_anchor=all prefill_ratio=0.75 prefill_prune_after_layer=8" \
#   "device=3 ckpt=QWEN task=mme limit=1000 majority_vote=5 temperature=1.0 \
#   enable_kdvz=True kdvz_ratio=0.75 \
#   enable_kd_prefill=True prefill_anchor=all prefill_ratio=0.75 prefill_prune_after_layer=8" \



)


bg_pids=()
for exp in "${experiments[@]}"; do
  if [[ "$mode" == "con" ]]; then
    (
      unset_all
      eval "$exp"
      echo "=== Experiment: ${exp} ==="
      sleep 2
      run_one
    ) & bg_pids+=("$!")
  else
    (
      unset_all
      eval "$exp"
      echo "=== Experiment: ${exp} ==="
      sleep 2
      run_one
    )
  fi
done

if [[ "$mode" == "con" ]]; then
  echo "Waiting for ${#bg_pids[@]} concurrent runs to finish..."
  wait "${bg_pids[@]}"
fi

# bash examples/models/test.sh
