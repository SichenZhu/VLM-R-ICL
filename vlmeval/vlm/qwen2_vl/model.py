from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch
from transformers import BitsAndBytesConfig

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr

from termcolor import colored

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = True,
        save_raw_output: bool = True,
        output_dir: str = "./outputs",
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        MODEL_CLS = None

        if listinstr(['2.5', '2_5', 'qwen25', 'vl-rethinker'], model_path.lower()):
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            MODEL_CLS = Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            MODEL_CLS = Qwen2VLForConditionalGeneration
            self.processor = Qwen2VLProcessor.from_pretrained(model_path)
        if rank == 0:
            print(f"now testing.....{self.model_path}")
        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # # If only one process and GPU memory is less than 40GB
        # if '72b' in self.model_path.lower():
        #     self.model = MODEL_CLS.from_pretrained(
        #         model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
        #     )
        #     self.model.eval()
        # elif auto_split_flag():
        #     assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
        #     # Will Use All GPUs to run one model
        #     self.model = MODEL_CLS.from_pretrained(
        #         model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
        #     )
        # else:
        if '72b' in self.model_path.lower():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            if rank == 0:
                print(model_path, " is quantized.")
        else:
            quantization_config = None
        
        self.model = MODEL_CLS.from_pretrained(
            model_path, torch_dtype=torch.float16, 
            device_map=rank, 
            attn_implementation='flash_attention_2', 
            quantization_config=quantization_config
        )
        self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': ensure_video_url(s['value']),
                    'min_pixels': self.min_pixels,
                    'max_pixels': self.max_pixels
                }
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'answer':
                item = {'type': 'answer', 'answer': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info, vision_process
            vision_process.MAX_RATIO = 300
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        structured_message = self._prepare_content(message, dataset=dataset)
        demo_message = []
        for s in structured_message:
            if s['type'] != 'answer':
                demo_message.append(s)
            else:
                assert len(demo_message) > 0, "Answer message should be the last one"
                messages.append({'role': 'user', 'content': demo_message})
                messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': s['answer']}]})
                demo_message = []
        if len(demo_message) > 0: # last one is query message
            messages.append({'role': 'user', 'content': demo_message})
            if listinstr(['vl-rethinker'], self.model_path.lower()):
                # need to append 
                # "\n\nPlease reason step by step, and put your final answer within \\boxed{} after the use queries."
                # after the use of queries
                messages.append({'role': 'user', 'content': "\n\nPlease reason step by step, and put your final answer within \\boxed{} after the use queries."})   
                # if self.verbose:
                #     print("vl-rethinker: ", end=">>>>>>>>>")
        
        if self.verbose:
            print('\033generate_inner[31m', end="")
            print(colored(f'{messages}', "dark_grey"), end="")
            print('\033[0m')

        if messages[-1]['role'] == 'assistant':
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=False)
        else:
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        
        resp = response.split('\\boxed{')[-1]
        lt = len(resp)
        counter, end = 1, None
        for i in range(lt):
            if resp[i] == '{':
                counter += 1
            elif resp[i] == '}':
                counter -= 1
            if counter == 0:
                end = i
                break
            elif i == lt - 1:
                end = lt
                break
        if end is not None:
            prediction = resp[:end]
        
        if self.post_process:
            response = prediction

        if self.verbose:
            print("\033generate_inner[32m", end="")
            print(colored(f'{response}', "blue"), end="")
            print("\033[0m")
        # return response
        return {"prediction": prediction, "rationale": response}
    
    def call_inner(self, message, dataset=None, output_hidden_states=False, output_attentions=False):
        try:
            from qwen_vl_utils import process_vision_info, vision_process
            vision_process.MAX_RATIO = 300
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        structured_message = self._prepare_content(message, dataset=dataset)
        demo_message = []
        for s in structured_message:
            if s['type'] != 'answer':
                demo_message.append(s)
            else:
                assert len(demo_message) > 0, "Answer message should be the last one"
                messages.append({'role': 'user', 'content': demo_message})
                messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': s['answer']}]})
                demo_message = []
        if len(demo_message) > 0:
            messages.append({'role': 'user', 'content': demo_message})

        if self.verbose:
            print("\033call_inner[31m", end="")
            print(colored(f'{messages}', "light_grey"), end="")
            print("\033[0m")

        if messages[-1]['role'] == 'assistant':
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=False)
        else:
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        outputs = self.model(
            **inputs,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return outputs
