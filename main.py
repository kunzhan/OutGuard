import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from datetime import datetime
import logging
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import precision_recall_curve, auc, roc_curve, precision_recall_fscore_support, confusion_matrix
from data_loader import load_processed_data, load_balanced_data, load_split_data_1, load_split_data_2, load_feat
from OutGuard import OutGuard, bag_contrastive_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_log_path(log_dir=f"logs"):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now()
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    filename = f"{script_name}_{now.strftime('%H_%M_%Y_%m_%d')}.log"
    return os.path.join(log_dir, filename)


def setup_logger(save=False, log_path=None):
    logger = logging.getLogger("OutGuard")
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


class MilBagDataset(Dataset):
    def __init__(self, bags, labels, selected_dim=25, dtype=torch.float32):
        assert len(bags) == len(labels)
        self.bags = []
        for b in bags:
            if isinstance(b, list) or isinstance(b, tuple):
                inst_tensors = []
                for inst in b:
                    t = torch.as_tensor(inst[f'layer_{selected_dim}'])
                    if t.dim() == 2 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    inst_tensors.append(t)
                bag_tensor = torch.cat(inst_tensors, dim=0).to(dtype)
                
            elif isinstance(b, dict):
                inst_tensors = []
                for inst_id, inst in b.items():
                    t = torch.as_tensor(inst[f'layer_{selected_dim}'])
                    if t.dim() == 2 and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.dim() == 1:
                        t = t.unsqueeze(0)
                    inst_tensors.append(t)
                bag_tensor = torch.cat(inst_tensors, dim=0).to(dtype)
            self.bags.append(bag_tensor)

        self.labels = torch.tensor(labels, dtype=dtype)

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]


def mil_collate_fn(batch):
    bags, labels = zip(*batch)
    padded_bags = pad_sequence(bags, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([b.shape[0] for b in bags], dtype=torch.long)
    max_len = padded_bags.shape[1]
    mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    labels = torch.stack(labels).unsqueeze(1)
    return padded_bags, labels, mask



def train_OutGuard(
    micl_model, train_safety_bags, train_unsafety_bags, val_safety_bags, val_unsafety_bags, 
    selected_dim=25, epochs=100, lr=1e-3, batch_size=32, lambda_contrastive=0.1, temperature=0.5,
    top_k_ratio=0.3, patience=20, device="cuda"):

    # Data preparation
    train_labels_pos = [0] * len(train_safety_bags)
    train_labels_neg = [1] * len(train_unsafety_bags)
    all_train_bags = train_safety_bags + train_unsafety_bags
    all_train_labels = train_labels_pos + train_labels_neg
    full_train_dataset = MilBagDataset(all_train_bags, all_train_labels, selected_dim=selected_dim)
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mil_collate_fn)
    
    val_labels_pos = [0] * len(val_safety_bags)
    val_labels_neg = [1] * len(val_unsafety_bags)
    all_val_bags = val_safety_bags + val_unsafety_bags
    all_val_labels = val_labels_pos + val_labels_neg
    val_dataset = MilBagDataset(all_val_bags, all_val_labels, selected_dim=selected_dim)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=mil_collate_fn)

    # 2. Model and Optimizer Settings
    micl_model.to(device)
    optimizer = optim.AdamW(micl_model.parameters(), lr=lr, weight_decay=1e-2)
    bce_criterion = nn.BCEWithLogitsLoss()
    
    # Early Stopping State
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(micl_model.state_dict())
    trigger_times = 0

    # 3. Training loop
    for epoch in range(epochs):
        # --- TRAIN PHASE ---
        micl_model.train()
        train_loss = 0.0
        train_loss_cls = 0.0
        train_loss_contrastive = 0.0
        # train_loss_sparse = 0.0
        train_total = 0
        
        for bags, labels, mask in train_loader:
            bags, labels, mask = bags.to(device), labels.to(device), mask.to(device)

            optimizer.zero_grad()
            logits, A, Z = micl_model(bags, mask, return_projection=True)
            loss_cls = bce_criterion(logits, labels)
            loss_contra = bag_contrastive_loss(
                Z, A, mask, temperature=temperature, top_k_ratio=top_k_ratio
            )
            A_valid = A.squeeze(-1) * mask.float()
            A_sum = A_valid.sum(dim=1, keepdim=True) + 1e-12
            A_norm = A_valid / A_sum
            loss = loss_cls + lambda_contrastive * loss_contra
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bags.size(0)
            train_loss_cls += loss_cls.item() * bags.size(0)
            train_loss_contrastive += loss_contra.item() * bags.size(0)
            train_total += labels.size(0)

        avg_train_loss = train_loss / train_total
        avg_train_cls = train_loss_cls / train_total
        avg_train_contra = train_loss_contrastive / train_total

        # --- VALIDATION PHASE ---
        micl_model.eval()
        val_loss = 0.0
        val_total = 0
        
        with torch.no_grad():
            for bags, labels, mask in val_loader:
                bags, labels, mask = bags.to(device), labels.to(device), mask.to(device)
                
                logits, A = micl_model(bags, mask, return_projection=False)
                loss = bce_criterion(logits, labels)
                
                val_loss += loss.item() * bags.size(0)
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train | Total: {avg_train_loss:.4f} | Cls: {avg_train_cls:.4f} | "
              f"Contra(Intra-bag): {avg_train_contra:.4f}")
        print(f"  Val   | Loss: {avg_val_loss:.4f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(micl_model.state_dict())
            trigger_times = 0
            print(f"  [Info] Best Val Loss updated: {best_val_loss:.4f}")
        else:
            trigger_times += 1
            print(f"  [Info] EarlyStopping counter: {trigger_times} out of {patience}")
            
            if trigger_times >= patience:
                print(f"  [Stop] Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}")
    micl_model.load_state_dict(best_model_wts)
    
    return micl_model


def test_OutGuard(micl_model, safety_bags, unsafety_bags, selected_dim=25, batch_size=32, device="cuda", dtype=torch.float32):
    
    labels_pos = [0] * len(safety_bags)
    labels_neg = [1] * len(unsafety_bags)
    
    all_bags = safety_bags + unsafety_bags
    all_labels = labels_pos + labels_neg
    
    dataset = MilBagDataset(all_bags, all_labels, selected_dim=selected_dim, dtype=dtype)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=mil_collate_fn)
    
    micl_model.eval()
    micl_model.to(device)
    
    y_true = []
    y_prob = []
    
    with torch.no_grad():
        for bags, labels, mask in tqdm(dataloader, desc="Test"):
            
            bags = bags.to(device)
            labels = labels.to(device)
            mask = mask.to(device)
            
            logits, _ = micl_model(bags, mask)
            probs = torch.sigmoid(logits)
            
            y_true.extend(labels.cpu().numpy().flatten())
            y_prob.extend(probs.cpu().numpy().flatten())

        
    return y_true, y_prob


def evaluate_AUPRC(true_labels, scores):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(true_labels, scores)
    auprc = auc(recall_arr, precision_arr)
    return auprc


def evaluate_AUROC(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    auroc = auc(fpr, tpr)
    return auroc


def danger(p, tau):
    return max(0.0, (p - tau) / (1 - tau))   


def evaluate_tpr_at_fpr(y_true, y_prob, target_fpr=0.05):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return tpr[idx], thresholds[idx]



def evaluation_metrics(y_true, y_prob_dict, threshold_dict):
    layer_preds = {}
    for layer, y_prob in y_prob_dict.items():
        y_prob = np.array(y_prob)
        threshold = float(threshold_dict[layer])
        preds = (y_prob >= threshold).astype(int)
        layer_preds[layer] = preds
        
    all_preds = np.stack(list(layer_preds.values()), axis=0)
    final_preds = (all_preds.sum(axis=0) >= (len(layer_preds) / 2)).astype(int) # vote
        
    AUPRC = evaluate_AUPRC(y_true, y_prob)
    AUROC = evaluate_AUROC(y_true, y_prob)
    
    # best_f1, best_threshold = 0.0, 0.0
    # for th in np.linspace(min(y_prob), max(y_prob), 100):
    #     preds = (y_prob >= th).astype(int)
    #     precision, recall, f1, _ = precision_recall_fscore_support(
    #         y_true, preds, average="binary", zero_division=1
    #     )
    #     if f1 > best_f1:
    #         best_f1, best_threshold = f1, th

    # final_preds = (y_prob >= best_threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        np.array(y_true), np.array(final_preds), average="binary", zero_division=1
    )
    tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()
    urup = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    arsp = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    tpr_at_5fpr, _ = evaluate_tpr_at_fpr(y_true, y_prob, target_fpr=0.05)
    
    return AUPRC, AUROC, precision, recall, f1, tpr_at_5fpr, urup, arsp
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="[OutGuard] Train & Choose Layers & Test")
    parser.add_argument("--base_model", '-m', type=str, default='llava_1_6')
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--choose_layer", action='store_true')
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--log", action='store_true')

    parser.add_argument("--device", '-d', type=str, default='cuda')
    parser.add_argument("--seed", '-s', type=int, default=42)

    parser.add_argument("--batch_size", '-b', type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--p0", type=float, default=0.9)
    parser.add_argument("--lambda_contrastive", type=float, default=0.1, help="Weight for intra-bag contrastive loss")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for contrastive loss")
    parser.add_argument("--top_k_ratio", type=float, default=0.3, help="Ratio of top-k instances to use as key instances")
    parser.add_argument("--attention_dim", type=int, default=256, help="Dimension of attention space")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    BASE_MODEL = args.base_model
    log_path = get_log_path(f"{BASE_MODEL}/logs")
    logger = setup_logger(save=args.log, log_path=log_path)
    
    logger.info("=" * 20)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"p0: {args.p0}")
    
    logger.info("=" * 20)
    logger.info("Dataset Summary:")
    try:
        dict_jailbreak_train, dict_jailbreak_val, dict_jailbreak_test, dict_benign_train, dict_benign_val, dict_benign_test = load_processed_data(logger, model=BASE_MODEL)
    except:
        dict_jailbreak, dict_benign = load_balanced_data(model=BASE_MODEL)
        dict_jailbreak_train_val, dict_jailbreak_test, dict_benign_train_val, dict_benign_test = load_split_data_1(dict_jailbreak, dict_benign, train_val_len=1000, model=BASE_MODEL)
        dict_jailbreak_train, dict_jailbreak_val, dict_benign_train, dict_benign_val = load_split_data_2(dict_jailbreak_train_val, dict_benign_train_val, model=BASE_MODEL)
        
        dict_jailbreak_train, dict_jailbreak_val, dict_jailbreak_test, dict_benign_train, dict_benign_val, dict_benign_test = load_processed_data(logger, model=BASE_MODEL)
    
    
    if args.train:

        logger.info("=" * 20)
        logger.info("Training Configuration:")
        logger.info(f"\tepochs: {args.epochs}")
        logger.info(f"\tbatch_size: {args.batch_size}")
        logger.info(f"\tlr: {args.lr}")
        logger.info(f"\tlambda_contrastive: {args.lambda_contrastive}")
        logger.info(f"\ttemperature: {args.temperature}")
        logger.info(f"\ttop_k_ratio: {args.top_k_ratio}")
        logger.info(f"\tattention_dim: {args.attention_dim}")
        logger.info("=" * 20)

        train_pos, train_neg = load_feat(dict_benign_train, model=BASE_MODEL), load_feat(dict_jailbreak_train, model=BASE_MODEL)
        val_pos, val_neg = load_feat(dict_benign_val, model=BASE_MODEL), load_feat(dict_jailbreak_val, model=BASE_MODEL)

        for i in range(33):
            
            if os.path.exists(f"{BASE_MODEL}/MICLs/{i}.pth"):
                logger.info(f"Model already exists: {BASE_MODEL}/MICLs/{i}.pth, skipping training.")
                continue
            
            micl = OutGuard(
                input_dim=4096,
                projection_dim=128
            ).to(device=args.device)
            
            micl = train_OutGuard(
                micl_model=micl, selected_dim=i, device=args.device,
                train_safety_bags=train_pos, train_unsafety_bags=train_neg, val_safety_bags=val_pos, val_unsafety_bags=val_neg,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                lambda_contrastive=args.lambda_contrastive, temperature=args.temperature, top_k_ratio=args.top_k_ratio,
            )
            
            os.makedirs(f"{BASE_MODEL}/MICLs", exist_ok=True)
            torch.save(micl.state_dict(), f"{BASE_MODEL}/MICLs/{i}.pth")
            logger.info(f"Model saved: {BASE_MODEL}/MICLs/{i}.pth")
            
    
    if args.choose_layer:
        
        chosen_csv_save_path = f"{BASE_MODEL}/choose_layer/MICLs.csv"
        try:
            threshold_record = {}
            with open(chosen_csv_save_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    threshold_record[row["layer"]] = float(row["best_threshold"])
            logger.info(f"Thresholds loaded from {chosen_csv_save_path}")
            
        except:
            attention_dim, lambda_contrastive, temperature, top_k_ratio, lr = args.attention_dim, args.lambda_contrastive, args.temperature, args.top_k_ratio, args.lr
            val_pos, val_neg = load_feat(dict_benign_val, model=BASE_MODEL), load_feat(dict_jailbreak_val, model=BASE_MODEL)

            if not os.path.exists(chosen_csv_save_path):
                os.makedirs(os.path.dirname(chosen_csv_save_path), exist_ok=True)
                with open(chosen_csv_save_path, 'w', encoding='utf-8') as f:
                    f.write("layer,best_threshold,AUPRC,AUROC,precision,recall,f1,tpr_at_5fpr,urup,arsp\n")
            
            threshold_record = {}
            for i in range(33):
                weights_path = f"{BASE_MODEL}/MICLs/{i}.pth"

                micl = OutGuard(
                    input_dim=4096, 
                    projection_dim=128
                ).to(device=args.device)
                state_dict = torch.load(weights_path, map_location=args.device)
                micl.load_state_dict(state_dict)
                
                y_true, y_prob = test_OutGuard(
                    micl_model=micl,
                    safety_bags=val_pos,
                    unsafety_bags=val_neg,
                    selected_dim=i,
                    batch_size=args.batch_size,
                    device=args.device
                )
                
                y_prob = np.array(y_prob)
                y_true = np.array(y_true)
                
                AUPRC = evaluate_AUPRC(y_true, y_prob)
                AUROC = evaluate_AUROC(y_true, y_prob)
                
                best_f1, best_threshold = 0.0, 0.0
                for th in np.linspace(min(y_prob), max(y_prob), 100):
                    preds = (y_prob >= th).astype(int)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, preds, average="binary", zero_division=1
                    )
                    if f1 > best_f1:
                        best_f1, best_threshold = f1, th            

                final_preds = (y_prob >= best_threshold).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, final_preds, average="binary", zero_division=1
                )
                tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()
                urup = fn / (tp + fn) if (tp + fn) > 0 else 0.0
                arsp = fp / (tn + fp) if (tn + fp) > 0 else 0.0
                tpr_at_5fpr, _ = evaluate_tpr_at_fpr(y_true, y_prob, target_fpr=0.05)


                threshold_record[str(i)] = best_threshold
                with open(chosen_csv_save_path, 'a', encoding='utf-8') as f:
                    f.write(f"{i},{best_threshold},{AUPRC:.4f},{AUROC:.4f},"
                        f"{precision:.4f},{recall:.4f},{f1:.4f},{tpr_at_5fpr:.4f},{urup:.4f},{arsp:.4f}\n")
                
                logger.info("=" * 20)
                logger.info(f"Layer: {i}")
                logger.info(f"\tAUPRC: {AUPRC:.4f}")
                logger.info(f"\tAUROC: {AUROC:.4f}")
                logger.info(f"\tbest_threshold: {best_threshold}")
                logger.info(f"\tprecision: {precision:.4f}")
                logger.info(f"\trecall: {recall:.4f}")
                logger.info(f"\tf1: {f1:.4f}")
                logger.info(f"\turup: {urup:.4f}")
                logger.info(f"\tarsp: {arsp:.4f}")
                logger.info(f"\ttpr_at_5fpr: {tpr_at_5fpr:.4f}")
                logger.info("=" * 20)
                
                
    
        
        chosen_dict_save_path = f"{BASE_MODEL}/choose_layer/Selected_MICLs.json"
        try:
            with open(chosen_dict_save_path, "r", encoding="utf-8") as f:
                chosen_dict = json.load(f)
            logger.info("=" * 20)
            logger.info(f"Chosen layers: {chosen_dict['layers']}")
            logger.info(f"Threshole: {chosen_dict['thresholds']}")
            logger.info(f"Scores: {chosen_dict['scores']}")
            logger.info("=" * 20)
        except:
            layers = [i+1 for i in range(32)]
            scores = []
            layer_chosen = []
            threshold_chosen = []
            scores_chosen = []
            
            try:
                print(f"attention_dim={attention_dim}, lambda_contrastive={lambda_contrastive}, temperature={temperature}, top_k_ratio={top_k_ratio}, lr={lr}")
                print(f"len_val_pos:{len(val_pos)}, len_val_neg={len(val_neg)}")
            except:
                attention_dim, lambda_contrastive, temperature, top_k_ratio, lr = args.attention_dim, args.lambda_contrastive, args.temperature, args.top_k_ratio, args.lr
                val_pos, val_neg = load_feat(dict_benign_val, model=BASE_MODEL), load_feat(dict_jailbreak_val, model=BASE_MODEL)
            
            for i in layers:
                weights_path = f"{BASE_MODEL}/MICLs/{i}.pth"            
                micl = OutGuard(
                    input_dim=4096, 
                    projection_dim=128
                ).to(device=args.device)
                state_dict = torch.load(weights_path, map_location=args.device)
                micl.load_state_dict(state_dict)
                
                y_true_pos, y_prob_pos = test_OutGuard(
                    micl_model=micl,
                    safety_bags=val_pos,
                    unsafety_bags=[],
                    selected_dim=i,
                    batch_size=args.batch_size,
                    device=args.device
                )
                y_true_neg, y_prob_neg = test_OutGuard(
                    micl_model=micl,
                    safety_bags=[],
                    unsafety_bags=val_neg,
                    selected_dim=i,
                    batch_size=args.batch_size,
                    device=args.device
                )
                num_pos = len(y_prob_pos)
                num_neg = len(y_prob_neg)
                
                
                threshold = float(threshold_record[str(i)])
                final_preds_pos = [int(p >= threshold) for p in y_prob_pos]
                final_preds_neg = [int(p >= threshold) for p in y_prob_neg]

                safe_danger = sum(danger(p, threshold) for p in y_prob_pos) / num_pos
                unsafe_danger = sum(danger(p, threshold) for p in y_prob_neg) / num_neg
                score = (1 - safe_danger + unsafe_danger) / 2
                scores.append(score)
                
                logger.info("=" * 20)
                logger.info(f"Layer {i}:")
                logger.info(f"\tThreshold: {threshold}")
                logger.info(f"\tSafe Danger Rate: {safe_danger:.4f}")
                logger.info(f"\tUnsafe Danger Rate: {unsafe_danger:.4f}")
                logger.info(f"\tScore: {score:.4f}")
                
                if score >= args.p0:
                    layer_chosen.append(i)
                    threshold_chosen.append(threshold)
                    scores_chosen.append(float(score))
                    
                else:
                    continue

            logger.info("=" * 20)
            logger.info(f"Chosen layers: {layer_chosen}")
            logger.info(f"Threshole: {threshold_chosen}")
            logger.info(f"Scores: {scores_chosen}")
            logger.info("=" * 20)
            
            with open(chosen_dict_save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "layers": layer_chosen,
                    "thresholds": threshold_chosen,
                    "scores": scores_chosen
                }, f, ensure_ascii=False, indent=4)


    if args.test_data is not None:
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
        
        
        attention_dim, lambda_contrastive, temperature, top_k_ratio, lr = args.attention_dim, args.lambda_contrastive, args.temperature, args.top_k_ratio, args.lr
        for i in layer_chosen:
            weights_path = f"{BASE_MODEL}/MICLs/{i}.pth" 
            assert os.path.exists(weights_path), f"Model weights not found at {weights_path}"
        
        if args.test_data != "test_set":
            dict_jailbreak_test, dict_benign_test = {}, {}
            test_data_list = args.test_data.split('+')
            for test_data in test_data_list:
                
                data_path = f"{BASE_MODEL}/instructions/{test_data}.json"
                with open(data_path, 'r') as f:
                    data_dict = json.load(f)

                dict_jailbreak_test[test_data] = {key: sample for key, sample in data_dict[test_data].items() if str(sample["toxicity"]) in ["1"]}
                dict_benign_test[test_data] = {key: sample for key, sample in data_dict[test_data].items() if str(sample["toxicity"]) in ["-2", "-1", "0"]}
                logger.info(f"Samples in {test_data}: Total ({len(data_dict[test_data])}), Benign ({len(dict_benign_test[test_data])}), Jailbreak ({len(dict_jailbreak_test[test_data])})")


        test_save_path = f"{BASE_MODEL}/result.csv"
        if not os.path.exists(test_save_path):
            os.makedirs(os.path.dirname(test_save_path), exist_ok=True)
            with open(test_save_path, 'w', encoding='utf-8') as f:
                f.write("Dataset,AUPRC,AUROC,precision,recall,f1,tpr_at_5fpr,urup,arsp\n")

        dataset_exists = False
        with open(test_save_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(f"{args.test_data},"):
                    dataset_exists = True
                    break

        if not dataset_exists:
            test_pos, test_neg = load_feat(dict_benign_test, model=BASE_MODEL), load_feat(dict_jailbreak_test, model=BASE_MODEL)

            y_prob_dict = {}
            for i in layer_chosen:
                weights_path = f"{BASE_MODEL}/MICLs/{i}.pth"

                micl = OutGuard(
                    input_dim=4096, 
                    projection_dim=128
                ).to(device=args.device)
                state_dict = torch.load(weights_path, map_location=args.device)
                micl.load_state_dict(state_dict)

                y_true, y_prob_dict[str(i)] = test_OutGuard(
                    micl_model=micl,
                    safety_bags=test_pos,
                    unsafety_bags=test_neg,
                    selected_dim=i,
                    batch_size=args.batch_size,
                    device=args.device
                )

            AUPRC, AUROC, precision, recall, f1, tpr_at_5fpr, urup, arsp = evaluation_metrics(y_true, y_prob_dict, threshold_dict)

            with open(test_save_path, 'a', encoding='utf-8') as f:
                f.write(f"{args.test_data},{AUPRC:.4f},{AUROC:.4f},"
                        f"{precision:.4f},{recall:.4f},{f1:.4f},{tpr_at_5fpr:.4f},{urup:.4f},{arsp:.4f}\n")

            logger.info("=" * 20)
            logger.info(f"\tAUPRC: {AUPRC:.4f}")
            logger.info(f"\tAUROC: {AUROC:.4f}")
            logger.info(f"\tprecision: {precision:.4f}")
            logger.info(f"\trecall: {recall:.4f}")
            logger.info(f"\tf1: {f1:.4f}")
            logger.info(f"\turup: {urup:.4f}")
            logger.info(f"\tarsp: {arsp:.4f}")
            logger.info(f"\ttpr_at_5fpr: {tpr_at_5fpr:.4f}")
            logger.info("=" * 20)
        else:
            logger.info(f"Results for dataset '{args.test_data}' already exist in {test_save_path}, skipping testing.")


    if args.log:            
        logger.info(f"\nLog file: {log_path}")