import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from io import BytesIO
import base64
from mimetypes import guess_type
import re

from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE, DATASET_MODALITY


class KimiVL(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="moonshotai/Kimi-VL-A3B-Thinking", **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            attn_implementation='flash_attention_2',
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.temperature = kwargs.get('temperature', None)

    def encode_image(self, image_path):
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        image_format = mime_type.split("/")[-1].upper() if mime_type else "JPEG"
        image = Image.open(image_path)
        # Handle the alpha channel
        if image.mode == "RGBA":
            image = self._rgba_to_rgb(image)

        encoded_image = self._encode_image(image, image_format)

        return encoded_image

    def _encode_image(self, image, image_format):
        with BytesIO() as output:
            image.convert("RGB").save(output, format=image_format)
            base64_encoded_data = base64.b64encode(output.getvalue()).decode("utf-8")
        return base64_encoded_data

    @staticmethod
    def _rgba_to_rgb(image):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")

    def message_to_promptimg(self, message, dataset=None):
        processed_message = []
        images = []
        for item in message:
            if item['type'] == 'text':
                processed_message.append({
                    "type": "text",
                    "text": f"{item['value']}"
                })
            elif item['type'] == 'image':
                image_path = item['value']
                encoded_image = self.encode_image(image_path)
                image = Image.open(BytesIO(base64.b64decode(encoded_image)))
                image.load()
                processed_message.append({
                    "type": "image",
                    "image": image_path,
                })
                images.append(image)
            elif item['type'] == 'answer':
                processed_message.append({
                    "type": "answer",
                    "answer": f"{item['value']}"
                })
        return processed_message, images

    def generate_inner(self, message, dataset=None):
        def extract_answer_content(output_str):
            # Try to find the content after **Answer: xxx** or **Answer:** xxx
            answer_pattern = (
                r"\*\*Answer(?::)?\*\*\s*(.*)"             # case 1: **Answer:** xxx
                r"|"
                r"\*\*Answer:\s*(.*?)\*\*"                 # case 2: **Answer: xxx**
                r"|"
                r"(?<!\*)\bAnswer:\s*(.+)"                 # case 3: plain Answer: xxx, not part of **
            )
            match = re.search(answer_pattern, output_str, re.DOTALL)

            if match:
                return (match.group(1) or match.group(2) or match.group(3)).strip()
            return output_str

        def replace_last_dot(input_string):
            if input_string.endswith("."):
                return input_string[:-1]
            else:
                return input_string
        
        structured_message, images = self.message_to_promptimg(message, dataset=dataset)
        messages = []
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

        print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(
            images=images, text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096, temperature=self.temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_output = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        response = extract_answer_content(raw_output)
        response = replace_last_dot(response)
        return {"prediction": response, "rationale": raw_output}
    
    def call_inner(self, message, dataset=None, output_hidden_states=False, output_attentions=False):
        structured_message, images = self.message_to_promptimg(message, dataset=dataset)
        messages = []
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

        print(f'\033[31m{messages}\033[0m')

        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = self.processor(
            images=images, text=text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        return outputs
