import os
import sys
import json
import argparse
import torch
from PIL import Image
from io import BytesIO
import requests
import re
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

def find_conv_mode(model_name):
    # select conversation mode based on the model name
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"  
    return conv_mode    


def adjust_query_for_images(qs, model):   
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    return qs


def construct_conv_prompt(sample, model_name, model):        
    conv = conv_templates[find_conv_mode(model_name)].copy()  
    if (sample.get('image') != None):     
        qs = adjust_query_for_images(sample['question'], model)
    else:
        qs = sample['question']
    conv.append_message(conv.roles[0], qs)  
    conv.append_message(conv.roles[1], None)       
    prompt = conv.get_prompt()
    return prompt


def load_image_from_path_or_url(image_file):
    if image_file is None:
        return None
    if isinstance(image_file, bytes):
        try:
            image = Image.open(BytesIO(image_file)).convert("RGB")
            return image
        except Exception as e:
            print(f"[WARN] load image from bytes failed: {e}")
            return None
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file, timeout=20)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_field):
    """
    image_field may be:
     - comma-separated paths string
     - list of bytes
     - single bytes
     - single path
    returns list of PIL images (or empty list)
    """
    out = []
    if image_field is None:
        return out
    if isinstance(image_field, str):
        # treat as comma-separated paths/urls
        for part in image_field.split(","):
            try:
                img = load_image_from_path_or_url(part.strip())
            except Exception as e:
                print(f"[WARN] failed to load image {part}: {e}")
                img = None
            if img is not None:
                out.append(img)
    elif isinstance(image_field, bytes):
        img = load_image_from_path_or_url(image_field)
        if img is not None:
            out.append(img)
    elif isinstance(image_field, list):
        for el in image_field:
            try:
                img = load_image_from_path_or_url(el)
            except Exception as e:
                print(f"[WARN] failed to load image element: {e}")
                img = None
            if img is not None:
                out.append(img)
    else:
        # treat as single path-like
        try:
            img = load_image_from_path_or_url(image_field)
            if img is not None:
                out.append(img)
        except Exception as e:
            print(f"[WARN] unsupported image field: {e}")
    return out


def extract_model_outputs(prompt, tokenizer, model, image_processor, images=None, output_hidden_states=True):
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )
    images_tensor = None
    images_size = None
    if images and len(images) > 0:
        images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
        images_size = [im.size for im in images]
        
    
    temperature = 0.2
    top_p = None
    num_beams = 1
    max_new_tokens = 4096
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=images_size,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_hidden_states=output_hidden_states
        )
        
    return outputs

 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default='cuda')
    parser.add_argument("--data", type=str, default="SafeBench")
    args = parser.parse_args()

    
    model_path = "liuhaotian/llava-v1.6-vicuna-7b"    
    model_name = get_model_name_from_path(model_path)        
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device_map=args.device,
        torch_dtype=torch.float16
    )
    
    dataset_dir = "llava_1_6/instructions"
    hidden_states_dir = "llava_1_6/HiddenStates"
    os.makedirs(hidden_states_dir, exist_ok=True)
    
    with open(f"{dataset_dir}/{args.data}.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
     
    for dataname in dataset.keys():
        for key, sample in tqdm(dataset[dataname].items(), desc=f"Processing {dataname}"):
        
            save_path = f"{hidden_states_dir}/{dataname}/{key}.pt"
            if os.path.exists(save_path) and sample.get('response') is not None:
                continue
            
            prompt = construct_conv_prompt(sample, model_name, model)
            images = load_images(sample.get('image', None))
            outputs = extract_model_outputs(prompt, tokenizer, model, image_processor, images=images, output_hidden_states=True)
            
            answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
            dataset[dataname][key]['response'] = answer
            dataset[dataname][key].pop('toxicity', None)
            dataset[dataname][key].pop('qwen_jailbreak_score', None)
            dataset[dataname][key].pop('deepseek_jailbreak_score', None)
            dataset[dataname][key].pop('gpt_jailbreak_score', None)
            dataset[dataname][key].pop('gemini_jailbreak_score', None)
            dataset[dataname][key].pop('grok_jailbreak_score', None)
            dataset[dataname][key].pop('len_token_ids', None)
            dataset[dataname][key].pop('len_hidden_states', None)
        
            with open(f"{dataset_dir}/{args.data}.json", 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            
            hidden_states_dict = {}
            for out_token_id, hidden_all in enumerate(outputs.hidden_states):
                hidden_states_dict[out_token_id] = {}
                
                for layer_id, tensor in enumerate(hidden_all):
                    hidden_states_dict[out_token_id][f"layer_{layer_id}"] = tensor[:, -1, :].cpu()
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(hidden_states_dict, save_path)