import os
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
import csv

import torch
from torch.utils.data import DataLoader

from OutGuard import OutGuard
from data_loader import load_processed_data
from main import MilBagDataset, mil_collate_fn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_log_path(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now()
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    filename = f"{script_name}_{now.strftime('%H_%M_%Y_%m_%d')}.log"
    return os.path.join(log_dir, filename)


def setup_logger(save=False, log_path=None):
    logger = logging.getLogger("MIL-Contrastive-Experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if save:
        assert log_path is not None, "`save=True` needs `log_path`"
        print(f"Logging to file: {log_path}")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



def test_OutGuard(micl_model, bags, selected_dim=25, batch_size=32, device="cuda", dtype=torch.float32):

    dataset = MilBagDataset(bags, [0]*len(bags), selected_dim=selected_dim, dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=mil_collate_fn)
    
    micl_model.eval()
    micl_model.to(device)
    
    y_prob = []
    
    with torch.no_grad():
        for bags, labels, mask in tqdm(dataloader, desc="Test"):
            
            bags = bags.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            logits, _ = micl_model(bags, mask)
            probs = torch.sigmoid(logits)
            
            y_prob.extend(probs.cpu().numpy().flatten())
    
    return y_prob


def final_vote(y_prob_dict, threshold_dict):
    layer_preds = {}
    for layer, y_prob in y_prob_dict.items():
        y_prob = np.array(y_prob)
        threshold = float(threshold_dict[layer])
        preds = (y_prob >= threshold).astype(int)
        layer_preds[layer] = preds
        
    all_preds = np.stack(list(layer_preds.values()), axis=0)
    final_preds = (all_preds.sum(axis=0) >= (len(layer_preds) / 2)).astype(int) # vote
    
    return final_preds.item()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="[OutGuard] Inference")
    parser.add_argument("--base_model", '-m', type=str, default='llava_1_6')
    parser.add_argument("--device", '-d', type=str, default='cuda')
    parser.add_argument("--seed", '-s', type=int, default=42)
    parser.add_argument("--test_data", default="test_set")
    parser.add_argument("--log", action='store_true')
    args = parser.parse_args()
    
    set_seed(args.seed)
    BASE_MODEL = args.base_model
    
    log_path = get_log_path(f"{BASE_MODEL}/logs")
    logger = setup_logger(save=args.log, log_path=log_path)
    
    logger.info("=" * 20)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    script_name = os.path.splitext(os.path.basename(__file__))[0]


    # Load data
    if args.test_data == "test_set":
        test_data = args.test_data
        logger.info("=" * 20)
        logger.info("Dataset Summary:")
        dict_jailbreak_train, dict_jailbreak_val, dict_jailbreak_test, dict_benign_train, dict_benign_val, dict_benign_test = load_processed_data(logger, model=BASE_MODEL)
        dict_test = {test_data: {}}
        for dataname, values in dict_benign_test.items():
            for key, sample in values.items():
                dict_test[test_data][f"{dataname}_{key}"] = sample
        for dataname, values in dict_jailbreak_test.items():
            for key, sample in values.items():
                dict_test[test_data][f"{dataname}_{key}"] = sample
    
    else:
        dict_test = {}
        test_data_list = args.test_data.split('+')
        for test_data in test_data_list:

            data_path = f"{BASE_MODEL}/instructions/{test_data}.json"
            with open(data_path, 'r') as f:
                data_dict = json.load(f)

            dict_test[test_data] = {key: sample for key, sample in data_dict[test_data].items()}
            logger.info(f"Samples in {test_data}: Total ({len(data_dict[test_data])})")


    if BASE_MODEL == "llava_1_6":
        from llava_1_6.response import get_model_name_from_path, load_pretrained_model, construct_conv_prompt
        from llava_1_6.response import load_images, extract_model_outputs
        model_path = "liuhaotian/llava-v1.6-vicuna-7b"    
        model_name = get_model_name_from_path(model_path)        
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device_map=args.device,
            torch_dtype=torch.float16
        )
        logger.info(f"Llava model loaded from {model_path}.")


    elif BASE_MODEL == "qwen_3_5":
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from qwen_3_5.response import load_images, extract_model_outputs
        model_path = "Qwen/Qwen3.5-9B"
        # model_path = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)
        logger.info(f"Qwen Model loaded from {model_path}.")

    else:
        raise ValueError(f"Unsupported base model: {BASE_MODEL}")


    # Check choosen layers
    chosen_dict_save_path = f"{BASE_MODEL}/choose_layer/Selected_MICLs.json"
    try:
        with open(chosen_dict_save_path, "r", encoding="utf-8") as f:
            chosen_dict = json.load(f)
        layer_chosen = chosen_dict["layers"]
        if len(layer_chosen) == 0:
            logger.error("No layers chosen for testing.")
            exit(0)
        threshold_dict = {str(layer_chosen[i]): float(chosen_dict["thresholds"][i]) for i in range(len(layer_chosen))}
        logger.info(f"Chosen layers for testing: {layer_chosen}")
    except FileNotFoundError:
        raise ValueError(f"'{chosen_dict_save_path}' not found.")
    except json.JSONDecodeError:
        raise ValueError(f"'{chosen_dict_save_path}' is not a valid JSON file.")
    

    torch.cuda.reset_peak_memory_stats(args.device)
    micl_models_dict = {}
    for i in layer_chosen:
        weights_path = f"{BASE_MODEL}/MICLs/{i}.pth" 
        assert os.path.exists(weights_path), f"Model weights not found at {weights_path}"

        m = OutGuard(input_dim=4096, projection_dim=128).to(args.device)
        m.load_state_dict(torch.load(weights_path, map_location=args.device))
        m.eval()
        micl_models_dict[str(i)] = m


    save_csv_path = f"{BASE_MODEL}/inference/{args.test_data}.csv"
    os.makedirs(f"{BASE_MODEL}/inference", exist_ok=True)


    existing_ids = set()
    if os.path.isfile(save_csv_path) and os.path.getsize(save_csv_path) > 0:
        with open(save_csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    existing_ids.add(row[0])

    
    file_exists = os.path.isfile(save_csv_path)
    file_empty = file_exists and os.path.getsize(save_csv_path) == 0


    for key in tqdm(dict_test[test_data].keys(), desc="infer & judge", unit="sample"):
        if key in existing_ids:
            print(f"Sample {key} already processed, skipping...")
            continue
        
        sample = dict_test[test_data][key]

        if BASE_MODEL == "llava_1_6":
            prompt = construct_conv_prompt(sample, model_name, model)
            images = load_images(sample.get('image', None))
            outputs = extract_model_outputs(
                prompt,
                tokenizer, model, image_processor,
                images=images,
                output_hidden_states=True
            )
            answer = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        elif BASE_MODEL == "qwen_3_5":
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
            answer = processor.decode(outputs['sequences'][0], skip_special_tokens=True).split("</think>")[1].strip()
        else:
            raise ValueError(f"Unsupported base model: {BASE_MODEL}")


        # extract feature
        hidden_states_dict = {}
        for out_token_id, hidden_all in enumerate(outputs.hidden_states):
            hidden_states_dict[out_token_id] = {}

            for layer_id, tensor in enumerate(hidden_all):
                hidden_states_dict[out_token_id][f"layer_{layer_id}"] = tensor[:, -1, :].to(args.device)


        y_prob_dict = {}
        for i in layer_chosen:
            y_prob_dict[str(i)] = test_OutGuard(
                micl_model=micl_models_dict[str(i)],
                bags=[hidden_states_dict],
                selected_dim=i,
                batch_size=20,
                device=args.device
            )
        toxicity = final_vote(y_prob_dict, threshold_dict)

        del hidden_states_dict; torch.cuda.empty_cache()

        with open(save_csv_path, "a", newline="", encoding="utf-8") as csvfile:
            csv_writer = csv.writer(csvfile)
            if not file_exists or file_empty:
                header = ["sample_id", "response", "toxicity"]
                csv_writer.writerow(header)
                file_empty = False
                file_exists = True
            row = [key, answer, toxicity]
            csv_writer.writerow(row)
        
    logger.info(f"Inference and judgment completed. Results saved to {save_csv_path}.")


            
    if args.log:            
        logger.info(f"\nLog file: {log_path}")
