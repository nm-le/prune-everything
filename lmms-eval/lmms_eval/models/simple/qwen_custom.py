import base64
import re
from io import BytesIO
from typing import List, Optional, Tuple, Union

import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.reasoning_model_utils import (
    parse_reasoning_model_answer,
)

from collections import Counter
import os
import json
import time



try:
    from transformers import QuantizedCacheConfig, QuantoQuantizedCache  # optional
    _has_quanto = True
except Exception:
    _has_quanto = False



try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen_custom")
class QwenCustom(lmms):
    """
    Basically Qwen2.5_VL Model
    "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct"
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache=True,
        attn_implementation: Optional[str] = None,
        min_pixels: int = 1280 * 28 * 28,
        max_pixels: int = 2048 * 28 * 28,
        max_num_frames: int = 32,
        use_custom_video_loader: Optional[bool] = False,
        fps: Optional[float] = None,  # Only applicable if use_custom_video_loader is True
        max_image_size: Optional[int] = None,  # Only applicable if use_custom_video_loader is True
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        reasoning_prompt: Optional[str] = None,
        
        benchmark: Optional[str] = None,
        enable_visionzip: Optional[bool] = False,
        visionzip_ratio: Optional[float] = 1.0,
        enable_kdvz: Optional[bool] = False,
        kdvz_ratio: Optional[float] = 1.0,
        enable_kd_prefill: Optional[bool] = False,
        prefill_anchor: Optional[str] = "all",
        prefill_ratio: Optional[float] = 0.0,
        prefill_prune_after_layer: Optional[int] = 0,
        enable_kd_decode: Optional[bool] = False,
        decode_anchor: Optional[str] = "all",
        decode_ratio: Optional[float] = 0.0,
        decode_prune_window: Optional[int] = 0,
        decode_prune_after_layer: Optional[int] = 0,
        
        majority_vote: Optional[int] = 1,
        temperature: Optional[float] = 0.0,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        # Validate attention implementation
        # valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        valid_attn_implementations = ["flash_attention_2"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        self.use_custom_video_loader = use_custom_video_loader
        self.fps = fps
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = max_image_size
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        self.accelerator = accelerator
        self._device = accelerator.device
        self.device_map = None  
        # if accelerator.num_processes > 1:
        #     self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        #     self.device_map = f"cuda:{accelerator.local_process_index}"
        # else:
        #     self._device = torch.device(device)
        #     self.device_map = device_map if device_map else device


        # Prepare model loading arguments
        model_kwargs = {
            "torch_dtype": "bfloat16",
            # "device_map": self.device_map,
        }

        # Add attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation



        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, **model_kwargs).to(self._device).eval()
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames

        # Minh 8/10/25
        self.enable_visionzip = enable_visionzip
        self.visionzip_ratio = visionzip_ratio
        self.enable_kdvz = enable_kdvz
        self.kdvz_ratio = kdvz_ratio
        self.enable_kd_prefill = enable_kd_prefill
        self.prefill_anchor = prefill_anchor
        self.prefill_ratio = prefill_ratio
        self.prefill_prune_after_layer = prefill_prune_after_layer
        self.enable_kd_decode = enable_kd_decode
        self.decode_anchor = decode_anchor
        self.decode_ratio = decode_ratio
        self.decode_prune_window = decode_prune_window
        self.decode_prune_after_layer = decode_prune_after_layer
        self.majority_vote = majority_vote
        self.temperature = temperature

        self.benchmark = benchmark

        # Minh: for logging
        # add date time prefix to log file path
        # datetime format: YYYYMMDD
        cur_datetime = time.strftime("%Y%m%d", time.localtime())
        self.log_file_path = os.path.join(output_path, pretrained.replace('/', '__'), f"{cur_datetime}_log.jsonl")
        # delete and create a new file if it already exists
        if os.path.exists(self.log_file_path):
            os.remove(self.log_file_path)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        if benchmark in ['mmstar']:
        # reasoning_prompt = " Please reason step by step, and put your final answer within \\boxed{}"
            reasoning_prompt = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Make sure your thinking process is long, at least 100 words, detailing your thought process and contemplating really deeply about the problem. Answer with only one letter."
        elif benchmark in ['mme']:
            reasoning_prompt = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Make sure your thinking process is long, at least 100 words, detailing your thought process and contemplating really deeply about the problem. Answer with only 'yes' or 'no'."
        elif benchmark in ['pope']:
            reasoning_prompt = " First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Answer with only one word."


        else:
            reasoning_prompt = " First output the thinking process in <think> </think> tags and then output the answer in <answer> </answer> tags."
            
        if reasoning_prompt:
            self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n")
        else:
            self.reasoning_prompt = None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)

        ip = self.processor.image_processor

        # Make sure we actually resize
        # print(ip)


        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals

        self._config = self.model.config
        self._max_length = kwargs.get("max_length", 2048)
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2.5_VL")


    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visual_list = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            gen_kwargs = all_gen_kwargs[0]

            # Set default until or update values from gen_kwargs if present
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])


            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            # Avoid using '\n\n' as a stopper for Qwen2.5VL to prevent truncation, which can lead to incorrect results
            until = [item for item in until if item != "\n\n"]


            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")

            batched_messages = []
            for i, context in enumerate(contexts):
                if "<image>" in context:
                    context = context.replace("<image>", "")

                message = [{"role": "system", "content": self.system_prompt}]
                # context = context.replace('\nAnswer the question using a single word or phrase.', ' Please answer yes or no.') ## MME specific hack 
                if self.reasoning_prompt:
                    context = context.strip() + self.reasoning_prompt
                    contexts[i] = context

                processed_visuals = []
                if visual_list[i] is not None:
                    for visual in visual_list[i]:
                        if isinstance(visual, str) and visual.endswith((".mp4", ".avi", ".mov")):  # Video file
                            vr = decord.VideoReader(visual)
                            first_frame = vr[0].asnumpy()
                            height, width = first_frame.shape[:2]
                            # max_pixels = height * width
                            processed_visuals.append({"type": "video", "video": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
                        elif isinstance(visual, Image.Image):  # Handle both single and multiple images
                            base64_image = visual.convert("RGB")
                            buffer = BytesIO()
                            base64_image.save(buffer, format="JPEG")
                            base64_bytes = base64.b64encode(buffer.getvalue())
                            base64_string = base64_bytes.decode("utf-8")
                            black = Image.new("RGB", (28, 28), 0)
                            # processed_visuals.append({"type": "image", "image": self.pil_to_data_uri(black, fmt="PNG")})

                            processed_visuals.append({"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})

                if self.interleave_visuals is False:
                    # context += " Please reason step by step, and put your final answer within \\boxed{}" 
                    message.append(
                        {
                            "role": "user",
                            "content": processed_visuals + [{"type": "text", "text": context}],
                        }
                    )
                    
                else:  # currently support find <image x> in the context
                    image_placeholders = re.findall(r"<image \d+>", context)
                    content_parts = []
                    text_parts = re.split(r"<image \d+>", context)
                    if text_parts[0]:
                        content_parts.append({"type": "text", "text": text_parts[0]})

                    for i, placeholder in enumerate(image_placeholders):
                        img_idx = int(re.search(r"<image (\d+)>", placeholder).group(1)) - 1
                        image_idx = min(img_idx, len(processed_visuals) - 1) if processed_visuals else 0
                        if processed_visuals and image_idx < len(processed_visuals):
                            content_parts.append(processed_visuals[image_idx])
                        if i + 1 < len(text_parts) and text_parts[i + 1]:
                            content_parts.append({"type": "text", "text": text_parts[i + 1]})
                    
                    message.append(
                        {
                            "role": "user",
                            "content": content_parts,
                        }
                    )

                batched_messages.append(message)

            texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batched_messages]

            image_inputs, video_inputs = process_vision_info(batched_messages)
            if video_inputs is not None:
                total_frames = video_inputs[0].shape[0]
                indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
                # Ensure unique indices if linspace produces duplicates for few frames
                indices = np.unique(indices)
                # Append the last frame index if not already included
                if total_frames - 1 not in indices:
                    indices = np.append(indices, total_frames - 1)
                    indices = np.unique(indices)  # Ensure uniqueness again
                video_inputs[0] = video_inputs[0][indices]
            inputs = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", do_resize=False)
            
            model_device = next(self.model.parameters()).device
            
            if hasattr(inputs, "to"):
                inputs = inputs.to(model_device)
            else:
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(model_device)

            # if self.device_map == "auto":
            #     inputs = inputs.to("cuda")
            # else:
            #     inputs = inputs.to(self.device)
            # Set default generation kwargs
            default_gen_kwargs = {
                "max_new_tokens": 1024,
                "temperature": 0.0,  # Set to 0 for greedy default
                "top_p": 0.9,
                "num_beams": 1,
            }
            # Update with provided kwargs
            current_gen_kwargs = {**default_gen_kwargs, **gen_kwargs}
            pad_token_id = self.tokenizer.pad_token_id

            current_gen_kwargs["temperature"] = self.temperature

            if current_gen_kwargs["temperature"] > 0:
                current_gen_kwargs["do_sample"] = True
                current_gen_kwargs["top_p"] = 0.9 ## TODO: revert back to 0.9
            else:
                current_gen_kwargs["do_sample"] = False
                current_gen_kwargs["temperature"] = None
                current_gen_kwargs["top_p"] = None

            current_gen_kwargs["max_new_tokens"] = 1024 # 1024 & 2048 for VL-Rethinker (msybe let's check eval for vl rethinker)
            print("current_gen_kwargs", current_gen_kwargs)


            vote_counters = [Counter() for _ in range(len(contexts))]
            reasoning_answers = [[] for _ in range(len(contexts))]
            generated_token_lens = []
            t0 = time.time()

            for _ in range(self.majority_vote):
                gen_kwargs_core = dict(
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=pad_token_id,
                    do_sample=current_gen_kwargs["do_sample"],
                    temperature=current_gen_kwargs["temperature"],
                    top_p=current_gen_kwargs["top_p"],
                    num_beams=current_gen_kwargs["num_beams"],
                    max_new_tokens=current_gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )

                # Minh 8/10/25 - add custom args for visionzip and keydiff
                custom_args = dict(
                    enable_visionzip=self.enable_visionzip,
                    visionzip_ratio=self.visionzip_ratio,
                    enable_kdvz=self.enable_kdvz,
                    kdvz_ratio=self.kdvz_ratio,
                    enable_kd_prefill=self.enable_kd_prefill,
                    prefill_anchor=self.prefill_anchor,
                    prefill_ratio=self.prefill_ratio,
                    prefill_prune_after_layer=self.prefill_prune_after_layer,
                    enable_kd_decode=self.enable_kd_decode,
                    decode_anchor=self.decode_anchor,
                    decode_ratio=self.decode_ratio,
                    decode_prune_window=self.decode_prune_window,
                    decode_prune_after_layer=self.decode_prune_after_layer,
                )

                cont = self.model.generate(**inputs, **gen_kwargs_core, **custom_args)


                generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
                generated_token_lens.append(generated_ids_trimmed[0].shape[0])
                answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

                for i, ans in enumerate(answers):
                    for term in until:
                        if len(term) > 0:
                            ans = ans.split(term)[0]
                    answers[i] = ans

                for ans, context in zip(answers, contexts):
                    print('reasoning ans', ans)
                    reasoning_answers[i].append(ans)
                    clean_ans = parse_reasoning_model_answer(ans)
                    if "answer:" in clean_ans.lower():
                        # remove trailing whitespaces and punctuations
                        clean_ans = clean_ans.rstrip(' \t\n\r.,!?;:')
                        # clean_ans will come in format "Final Answer:<ans>" - should only get the <ans> part
                        clean_ans = clean_ans.split("Final Answer:")[-1]
                        clean_ans = clean_ans.strip().lower()
                    clean_ans = clean_ans.strip().lower()
                    vote_counters[i][clean_ans] += 1

            elapsed = time.time() - t0
            for i, context in enumerate(contexts):
                print('all ans', vote_counters[i])
                final_ans = vote_counters[i].most_common(1)[0][0] if vote_counters[i] else ""
                print('voted ans', final_ans)
                res.append(final_ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), clean_ans)
                pbar.update(1)

                log_line = {
                    "voted_ans": final_ans,
                    "time_elapsed": elapsed,
                    "reasoning_answers": reasoning_answers[i],
                    "reasoning_token_lengths": generated_token_lens,
                    "reasoning_word_counts": [len(str(a).split()) for a in reasoning_answers[i]],
                    "vote_counters": dict(vote_counters[i]),
                }

                
                # Minh: append log line to log file
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write("{\n")
                    f.write(f'  "context": {json.dumps(context, ensure_ascii=False)},\n')
                    f.write(f'  "voted_ans": "{log_line["voted_ans"]}",\n')
                    f.write(f'  "time_elapsed": {log_line["time_elapsed"]},\n')
                    f.write(f'  "reasoning_answers": [\n')
                    for ans in log_line["reasoning_answers"]:
                        escaped = ans.rstrip("\n")
                        f.write("    " + json.dumps(escaped, ensure_ascii=False) + ",\n")
                    f.write("  ],\n")
                    f.write(f'  "reasoning_token_lengths": {json.dumps(log_line["reasoning_token_lengths"], ensure_ascii=False)},\n')
                    f.write(f'  "reasoning_word_counts": {json.dumps(log_line["reasoning_word_counts"], ensure_ascii=False)},\n')
                    f.write(f'  "vote_counters": {json.dumps(log_line["vote_counters"], ensure_ascii=False)}\n')
                    f.write("}\n\n")

        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
