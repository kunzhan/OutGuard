import os
import json
import argparse
import torch
from PIL import Image
from io import BytesIO
import requests
import re
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText


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


def extract_model_outputs(prompt, image, model, processor, output_hidden_states=True):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": prompt}
            ]
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=4096,
        return_dict_in_generate=True,
        output_hidden_states=output_hidden_states
    )
    
    return outputs




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default='cuda')
    parser.add_argument("--data", type=str, default="SafeBench")
    args = parser.parse_args()


    model_id = "Qwen/Qwen3.5-9B"
    # model_id = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model, error details: {e}")


    dataset_dir = "qwen_3_5/instructions"
    hidden_states_dir = "qwen_3_5/HiddenStates"
    os.makedirs(hidden_states_dir, exist_ok=True)

    with open(f"{dataset_dir}/{args.data}.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for dataname in dataset.keys():
        for key, sample in tqdm(dataset[dataname].items(), desc=f"Processing {dataname}"):

            save_path = f"{hidden_states_dir}/{dataname}/{key}.pt"
            if os.path.exists(save_path) and sample.get('response') is not None:
                continue

            prompt = sample["question"]
            images = load_images(sample.get('image', None))

            if len(images) == 0:
                continue

            outputs = extract_model_outputs(
                prompt,
                images[0] if len(images) > 0 else None,
                model,
                processor
            )
            answer = processor.decode(outputs['sequences'][0], skip_special_tokens=True).split("</think>")[1]
            token_ids = processor.tokenizer(answer).input_ids
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