import os
import json
import random
from tqdm import tqdm
import torch


def load_feat(input_dict, model="llava_1_6"):
    bag_instances_list = []

    for data_name in input_dict.keys():
        for key in tqdm(input_dict[data_name].keys(), desc=f"Loading features from {data_name}"):
            hidden_path = f"{model}/HiddenStates/{data_name}/{key}.pt"
            bag_instances = torch.load(hidden_path, map_location='cpu')
            bag_instances_list.append(bag_instances)

    return bag_instances_list


def load_processed_data(logger, model="llava_1_6"):
    
    with open(f"{model}/data/jailbreak_train.json", "r", encoding="utf-8") as f:
        jailbreak_train = json.load(f)
    with open(f"{model}/data/jailbreak_val.json", "r", encoding="utf-8") as f:
        jailbreak_val = json.load(f)
    with open(f"{model}/data/jailbreak_test.json", "r", encoding="utf-8") as f:
        jailbreak_test = json.load(f)
    with open(f"{model}/data/beni_train.json", "r", encoding="utf-8") as f:
        benign_train = json.load(f)
    with open(f"{model}/data/beni_val.json", "r", encoding="utf-8") as f:
        benign_val = json.load(f)
    with open(f"{model}/data/beni_test.json", "r", encoding="utf-8") as f:
        benign_test = json.load(f)
        
    len_jailbreak_train = sum(len(samples) for samples in jailbreak_train.values())
    len_jailbreak_val = sum(len(samples) for samples in jailbreak_val.values())
    len_jailbreak_test = sum(len(samples) for samples in jailbreak_test.values())
    len_jailbreak = len_jailbreak_train + len_jailbreak_val + len_jailbreak_test
    
    len_benign_train = sum(len(samples) for samples in benign_train.values())
    len_benign_val = sum(len(samples) for samples in benign_val.values())
    len_benign_test = sum(len(samples) for samples in benign_test.values())
    len_benign = len_benign_train + len_benign_val + len_benign_test
    
    logger.info(f"\tTotal samples: {len_benign + len_jailbreak} (Benign: {len_benign}, Jailbreak: {len_jailbreak})")
    logger.info(f"\tTrain samples: {len_jailbreak_train + len_benign_train} (Benign: {len_benign_train}, Jailbreak: {len_jailbreak_train})")
    logger.info(f"\tValidation samples: {len_jailbreak_val + len_benign_val} (Benign: {len_benign_val}, Jailbreak: {len_jailbreak_val})")
    logger.info(f"\tTest samples: {len_jailbreak_test + len_benign_test} (Benign: {len_jailbreak_test}, Jailbreak: {len_benign_test})")
    
    return jailbreak_train, jailbreak_val, jailbreak_test, benign_train, benign_val, benign_test





def load_balanced_data(limit=1347, model="llava_1_6"):
    '''
    Preprocess to balance samples while loading
    '''

    save_path_harm = f"{model}/data/dict_harm.json"
    save_path_beni = f"{model}/data/dict_beni.json"

    if os.path.exists(save_path_harm) and os.path.exists(save_path_beni):
        with open(save_path_harm, "r", encoding="utf-8") as f:
            final_dict_harm = json.load(f)
        with open(save_path_beni, "r", encoding="utf-8") as f:
            final_dict_beni = json.load(f)

    else:
        os.makedirs(f"{model}/data", exist_ok=True)
        with open(f"{model}/instructions/SafeBench.json", 'r', encoding='utf-8') as f:
            dict_safebench = json.load(f)
        with open(f"{model}/instructions/advbench.json", 'r', encoding='utf-8') as f:
            dict_advbench = json.load(f)
        with open(f"{model}/instructions/GQA.json", 'r', encoding='utf-8') as f:
            dict_gqa = json.load(f)
        len_gqa = len(dict_gqa["GQA"])
        len_safebench = len(dict_safebench["SafeBench"])
        len_advbench = len(dict_advbench["advbench"])

        final_dict_harm = {}
        final_dict_beni = {}

        
        final_dict_harm['advbench'] = {}
        keys_harm_advbench = list(key for key, sample in dict_advbench["advbench"].items() if str(sample["toxicity"]) == "1")[:len_advbench]
        for key in tqdm(keys_harm_advbench, desc=f"harm feature preparation from advbench"):
            final_dict_harm['advbench'][key] = dict_advbench["advbench"][key]
        print(f"Harmful samples in advbench: {len(final_dict_harm['advbench'])}")

        final_dict_harm['SafeBench'] = {}
        keys_harm_SafeBench = list(key for key, sample in dict_safebench["SafeBench"].items() if str(sample["toxicity"]) == "1")[:limit-len(keys_harm_advbench)]
        for key in tqdm(keys_harm_SafeBench, desc=f"harm feature preparation from SafeBench"):
            final_dict_harm['SafeBench'][key] = dict_safebench["SafeBench"][key]
        print(f"Harmful samples in SafeBench: {len(keys_harm_SafeBench)}")
        
        
        final_dict_beni['GQA'] = {}
        keys_benign_gqa = list(dict_gqa["GQA"].keys())[:len_gqa]
        for key in tqdm(keys_benign_gqa, desc=f"benign feature preparation from GQA"):
            final_dict_beni['GQA'][key] = dict_gqa["GQA"][key]
        print(f"Benign samples in GQA: {len(keys_benign_gqa)}")
        
        
        target_n = limit - len_gqa
        
        keys_tox_0_advbench = [key for key, sample in dict_advbench["advbench"].items() if str(sample["toxicity"]) == "0"]
        keys_tox_m1_advbench = [key for key, sample in dict_advbench["advbench"].items() if str(sample["toxicity"]) == "-1"]

        target_advbench = target_n // 2
        target_advbench_half = target_advbench // 2
        take_0_advbench = min(len(keys_tox_0_advbench), target_advbench_half)
        take_m1_advbench = min(len(keys_tox_m1_advbench), target_advbench_half)
        keys_benign_advbench = (
            keys_tox_0_advbench[:take_0_advbench] +
            keys_tox_m1_advbench[:take_m1_advbench]
        )
        
        final_dict_beni['advbench'] = {}
        for key in tqdm(keys_benign_advbench, desc=f"benign feature preparation from advbench"):
            final_dict_beni['advbench'][key] = dict_advbench["advbench"][key]
        print(f"Benign samples in advbench: {len(keys_benign_advbench)}")


        advbench_miss_id = 0 if take_0_advbench < take_m1_advbench else (-1 if take_m1_advbench < take_0_advbench else None)
        advbench_miss_num = take_m1_advbench - take_0_advbench if advbench_miss_id == 0 else (take_0_advbench - take_m1_advbench if advbench_miss_id == -1 else 0)
        
        
        keys_tox_0_safebench = [key for key, sample in dict_safebench["SafeBench"].items() if str(sample["toxicity"]) == "0"]
        keys_tox_m1_safebench = [key for key, sample in dict_safebench["SafeBench"].items() if str(sample["toxicity"]) == "-1"]
        
        target_safebench = target_n - len(keys_benign_advbench) - advbench_miss_num
        target_safebench_half = target_safebench // 2
        take_0_safebench = min(len(keys_tox_0_safebench), target_safebench_half)
        take_m1_safebench = min(len(keys_tox_m1_safebench), target_safebench_half)
        keys_benign_safebench = (
            keys_tox_0_safebench[:take_0_safebench] +
            keys_tox_m1_safebench[:take_m1_safebench]
        )
        cur_n = len(keys_benign_safebench)
        
        if cur_n < target_safebench:
            remaining = target_safebench - cur_n

            if len(keys_tox_0_safebench) - take_0_safebench > len(keys_tox_m1_safebench) - take_m1_safebench:
                keys_benign_safebench += keys_tox_0_safebench[take_0_safebench: take_0_safebench + remaining]
                take_0_safebench += remaining
            else:
                keys_benign_safebench += keys_tox_m1_safebench[take_m1_safebench: take_m1_safebench + remaining]
                take_m1_safebench += remaining
        
        if advbench_miss_num > 0:
            if advbench_miss_id == 0:
                keys_benign_safebench += keys_tox_0_safebench[take_0_safebench: take_0_safebench + advbench_miss_num]
                take_0_safebench += advbench_miss_num
            else:
                keys_benign_safebench += keys_tox_m1_safebench[take_m1_safebench: take_m1_safebench + advbench_miss_num]
                take_m1_safebench += advbench_miss_num
        
        final_dict_beni['SafeBench'] = {}
        for key in tqdm(keys_benign_safebench, desc=f"benign feature preparation from safebench"):
            final_dict_beni['SafeBench'][key] = dict_safebench["SafeBench"][key]
        print(f"Benign samples in SafeBench: {len(keys_benign_safebench)}")

        
        # cur_benign_n = len(bag_benign_list)
        cur_benign_n = 0
        cur_benign_n += sum(len(part) for part in final_dict_beni.values())
        if cur_benign_n < limit:
            remain_advbench_id = -1 if advbench_miss_id == 0 else (0 if advbench_miss_id == -1 else None)
            remain_advbench_num = limit - cur_benign_n
            
            if remain_advbench_id == 0:
                keys_benign_advbench_extra = keys_tox_0_advbench[take_0_advbench: take_0_advbench + remain_advbench_num]
            elif remain_advbench_id == -1:
                keys_benign_advbench_extra = keys_tox_m1_advbench[take_m1_advbench: take_m1_advbench + remain_advbench_num]
            else:
                raise ValueError("No remaining benign samples can be added.")
            
            for key in tqdm(keys_benign_advbench_extra, desc=f"extra benign feature preparation from advbench"):
                final_dict_beni['advbench'][key] = dict_advbench["advbench"][key]
            print(f"Benign samples in advbench: {len(keys_benign_advbench_extra)}")
                

        # cur_benign_n = len(final_dict_beni)
        cur_benign_n = 0
        cur_benign_n += sum(len(part) for part in final_dict_beni.values())
        if cur_benign_n < limit:
            remain_safebench_id = -1 if advbench_miss_id == 0 else (0 if advbench_miss_id == -1 else None)
            remain_safebench_num = limit - cur_benign_n
            
            if remain_safebench_id == 0:
                keys_benign_safebench_extra = keys_tox_0_safebench[take_0_safebench: take_0_safebench + remain_safebench_num]
            elif remain_safebench_id == -1:
                keys_benign_safebench_extra = keys_tox_m1_safebench[take_m1_safebench: take_m1_safebench + remain_safebench_num]
            else:
                raise ValueError("No remaining benign samples can be added.")
            
            for key in tqdm(keys_benign_safebench_extra, desc=f"extra benign feature preparation from SafeBench"):
                final_dict_beni['SafeBench'][key] = dict_safebench["SafeBench"][key]
            print(f"Benign samples in SafeBench: {len(keys_benign_safebench_extra)}")
            
        
        cur_harm_n = 0
        cur_harm_n += sum(len(part) for part in final_dict_harm.values())
        assert cur_harm_n == limit
        cur_benign_n = 0
        cur_benign_n += sum(len(part) for part in final_dict_beni.values())
        assert cur_benign_n == limit

        with open(f"{model}/data/dict_harm.json", "w", encoding="utf-8") as f:
            json.dump(final_dict_harm, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/dict_beni.json", "w", encoding="utf-8") as f:
            json.dump(final_dict_beni, f, ensure_ascii=False, indent=4)
    
    return final_dict_harm, final_dict_beni



def load_split_data_1(dict_jailbreak, dict_benign, model="llava_1_6", train_val_len=1000):
    '''
    Preprocess to split data into train+val and test sets
    '''

    try:
        with open(f"{model}/data/jailbreak_train_val.json", "r", encoding="utf-8") as f:
            dict_jailbreak_train_val = json.load(f)
        with open(f"{model}/data/jailbreak_test.json", "r", encoding="utf-8") as f:
            dict_jailbreak_test = json.load(f)
        with open(f"{model}/data/beni_train_val.json", "r", encoding="utf-8") as f:
            dict_benign_train_val = json.load(f)
        with open(f"{model}/data/beni_test.json", "r", encoding="utf-8") as f:
            dict_benign_test = json.load(f)

    except:
        print("Pre-split data not found, performing new split...")
        jail_items = []
        dict_jailbreak_temp = {}
        for dataname, sub_dict in dict_jailbreak.items():
            for key_id, value in sub_dict.items():
                new_key = f"{dataname}_{key_id}"
                dict_jailbreak_temp[new_key] = value
                jail_items.append((new_key, value))

        random.shuffle(jail_items)
        jail_train_val_items = jail_items[:train_val_len]
        jail_test_items = jail_items[train_val_len:]
        dict_jailbreak_train_val = dict(jail_train_val_items)
        dict_jailbreak_test = dict(jail_test_items)

        beni_items = []
        dict_benign_temp = {}
        for dataname, sub_dict in dict_benign.items():
            for key_id, value in sub_dict.items():
                new_key = f"{dataname}_{key_id}"
                dict_benign_temp[new_key] = value
                beni_items.append((new_key, value))

        random.shuffle(beni_items)
        beni_train_val_items = beni_items[:train_val_len]
        beni_test_items = beni_items[train_val_len:]
        dict_benign_train_val = dict(beni_train_val_items)
        dict_benign_test = dict(beni_test_items)

        jail_keys_train_val = {}
        for key in dict_jailbreak_train_val.keys():
            dataname, key_id = key.split("_")
            if dataname not in jail_keys_train_val:
                jail_keys_train_val[dataname] = []
            jail_keys_train_val[dataname].append(key_id)

        jail_keys_test = {}
        for key in dict_jailbreak_test.keys():
            dataname, key_id = key.split("_")
            if dataname not in jail_keys_test:
                jail_keys_test[dataname] = []
            jail_keys_test[dataname].append(key_id)
        beni_keys_train_val = {}
        for key in dict_benign_train_val.keys():
            dataname, key_id = key.split("_")
            if dataname not in beni_keys_train_val:
                beni_keys_train_val[dataname] = []
            beni_keys_train_val[dataname].append(key_id)
        beni_keys_test = {}
        for key in dict_benign_test.keys():
            dataname, key_id = key.split("_")
            if dataname not in beni_keys_test:
                beni_keys_test[dataname] = []
            beni_keys_test[dataname].append(key_id)

        dict_jailbreak_train_val = {}
        for dataname in jail_keys_train_val.keys():
            dict_jailbreak_train_val[dataname] = {}
            for key_id in jail_keys_train_val[dataname]:
                dict_jailbreak_train_val[dataname][key_id] = dict_jailbreak[dataname][key_id]
        dict_jailbreak_test = {}
        for dataname in jail_keys_test.keys():
            dict_jailbreak_test[dataname] = {}
            for key_id in jail_keys_test[dataname]:
                dict_jailbreak_test[dataname][key_id] = dict_jailbreak[dataname][key_id]
        dict_benign_train_val = {}
        for dataname in beni_keys_train_val.keys():
            dict_benign_train_val[dataname] = {}
            for key_id in beni_keys_train_val[dataname]:
                dict_benign_train_val[dataname][key_id] = dict_benign[dataname][key_id]
        dict_benign_test = {}
        for dataname in beni_keys_test.keys():
            dict_benign_test[dataname] = {}
            for key_id in beni_keys_test[dataname]:
                dict_benign_test[dataname][key_id] = dict_benign[dataname][key_id]
        
        with open(f"{model}/data/jailbreak_train_val.json", "w", encoding="utf-8") as f:
            json.dump(dict_jailbreak_train_val, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/jailbreak_test.json", "w", encoding="utf-8") as f:
            json.dump(dict_jailbreak_test, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/beni_train_val.json", "w", encoding="utf-8") as f:
            json.dump(dict_benign_train_val, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/beni_test.json", "w", encoding="utf-8") as f:
            json.dump(dict_benign_test, f, ensure_ascii=False, indent=4)

    return dict_jailbreak_train_val, dict_jailbreak_test, dict_benign_train_val, dict_benign_test


def split_train_val(dict_train_val, train_precent=0.8):
    all_samples = []
    for dataname, sub_dict in dict_train_val.items():
        for key_id in sub_dict.keys():
            all_samples.append((dataname, key_id))
    random.shuffle(all_samples)
    total_count = len(all_samples)
    split_idx = int(total_count * train_precent)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    train_dict, val_dict = {}, {}
    
    for d_name, k_id in train_samples:
        if d_name not in train_dict:
            train_dict[d_name] = {}
        train_dict[d_name][k_id] = dict_train_val[d_name][k_id]
        
    for d_name, k_id in val_samples:
        if d_name not in val_dict:
            val_dict[d_name] = {}
        val_dict[d_name][k_id] = dict_train_val[d_name][k_id]
        
    return train_dict, val_dict


def load_split_data_2(dict_jailbreak_train_val, dict_benign_train_val, model="llava_1_6"):

    '''
    Preprocess to split train+val data into train and val sets
    '''

    try:
        with open(f"{model}/data/jailbreak_train.json", "r", encoding="utf-8") as f:
            dict_jailbreak_train = json.load(f)
        with open(f"{model}/data/jailbreak_val.json", "r", encoding="utf-8") as f:
            dict_jailbreak_val = json.load(f)
        with open(f"{model}/data/beni_train.json", "r", encoding="utf-8") as f:
            dict_benign_train = json.load(f)
        with open(f"{model}/data/beni_val.json", "r", encoding="utf-8") as f:
            dict_benign_val = json.load(f)
    except:
        print("Pre-split train/val data not found, performing new split...")
        dict_jailbreak_train, dict_jailbreak_val = split_train_val(dict_jailbreak_train_val, train_precent=0.8)
        dict_benign_train, dict_benign_val = split_train_val(dict_benign_train_val, train_precent=0.8)
        with open(f"{model}/data/jailbreak_train.json", "w", encoding="utf-8") as f:
            json.dump(dict_jailbreak_train, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/jailbreak_val.json", "w", encoding="utf-8") as f:
            json.dump(dict_jailbreak_val, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/beni_train.json", "w", encoding="utf-8") as f:
            json.dump(dict_benign_train, f, ensure_ascii=False, indent=4)
        with open(f"{model}/data/beni_val.json", "w", encoding="utf-8") as f:
            json.dump(dict_benign_val, f, ensure_ascii=False, indent=4)
    return dict_jailbreak_train, dict_jailbreak_val, dict_benign_train, dict_benign_val
