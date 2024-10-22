import torch
from PIL import Image
import os.path as osp
import sys
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE


class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def split_model(self):
        import math
        device_map = {}
        num_gpus = torch.cuda.device_count()
        rank, world_size = get_rank_and_world_size()
        num_gpus = num_gpus // world_size
        num_layers = 80
        num_layers_per_gpu = num_layers // num_gpus
        num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
        layer_cnt = 0
        for i, num_layer in enumerate(num_layers_per_gpu):
            for j in range(num_layer):
                device_map[f'model.transformer.blocks.{layer_cnt}'] = rank + world_size * i
                layer_cnt += 1

        last_gpu = rank + world_size * (num_gpus - 1)
        device_map['model.transformer.wte'] = rank
        device_map['model.transformer.emb_drop'] = rank
        device_map['model.transformer.ln_f'] = rank
        device_map['model.transformer.ff_out'] = last_gpu
        device_map['model.vision_backbone'] = last_gpu
        return device_map

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            import einops
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        if '72b' not in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='cuda')
        else:
            device_map = self.split_model()
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=device_map)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.kwargs = kwargs
        self.model_name = model_path

    def generate_inner(self, message, dataset=None):
        from transformers import GenerationConfig
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        # process the image and text
        inputs = self.processor.process(images=[image], text=prompt)

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # print the generated text
        return generated_text
