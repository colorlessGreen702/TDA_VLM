import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator
import gc
import csv
import os
import timeit
import json

from transformers import CLIPProcessor, CLIPModel, HqqConfig, QuantoConfig, BitsAndBytesConfig
# from hqq.utils.patching import prepare_for_inference
# from hqq.core.quantize import *

# import clip
from utils import *


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true', help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/', help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True, help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).cuda().half()
        else:
            cache_values = (F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        image_features = image_features
        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits

def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        # pos_enabled, neg_enabled = False, False
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images ,clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)

            if pos_enabled:
                update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'], clip_weights, (neg_params['mask_threshold']['lower'], neg_params['mask_threshold']['upper']))

                
            acc = cls_acc(final_logits, target)  
            accuracies.append(acc)

            # if i%1000==0:
            #     print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)



# Function to write data to JSON file
def write_to_json(filename, data):
    # Read existing data from the file
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                file_data = []  # In case the file is empty or corrupted
    else:
        file_data = []

    updated = False
    for entry in file_data:
        if entry['method'] == data['method'] and entry['benchmark'] == data['benchmark']:
            # Update the existing entry
            entry.update(data)
            updated = True
            break

    # If no match was found, add the new entry
    if not updated:
        file_data.append(data)

    # Write updated data back to the file
    with open(filename, 'w') as file:
        json.dump(file_data, file, indent=4)
    print(f"Data written to {filename} successfully.")


def check_entry_exists(file_name, method, benchmark):
    try:
        # Read the existing data from the JSON file
        with open(file_name, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        # If the file does not exist, return False
        return False

    # Check if an entry with the same method and benchmark exists
    for entry in existing_data:
        if entry.get('method') == method and entry.get('benchmark') == benchmark:
            return True

    # Return False if no match is found
    return False


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    nbits_list = [3,2,1]
    group_sizes = [128, 64, 32, 16, 8]

    # datasets = [caltech101/dtd/oxford_pets/ucf101/A/V]

    cifar10c_corruption_types = ['original','gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 
                                 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 
                                 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 
                                 'jpeg_compression']


    for nb in nbits_list:
        for gp in group_sizes:

            quant_config = HqqConfig(nbits=nb, group_size=gp, quant_zero=False, quant_scale=False, axis=0)

            # if nb == 8:
            #     quant_config = BitsAndBytesConfig(load_in_8bit=True)
            # elif nb == 4:
            #     quant_config = BitsAndBytesConfig(load_in_4bit=True)

            # quantos = "int" + str(nb)
            # print(quantos)
            # quant_config = QuantoConfig(weights=quantos)

            # quant_config = None

            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16, device_map="cuda", quantization_config=quant_config)
            preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            # Set random seed
            random.seed(1)
            torch.manual_seed(1)
            
            # model.append(f'HQQ {nb}bit {gp}gp HF Prompts No TDA')
            # model.append(f'BnB {nb}bit HF Prompts No TDA')

            # Run TDA on each dataset
            datasets = args.datasets.split('/')
            for dataset_name in datasets:
                method = f'TDA HQQ {nb}bit {gp}gp'

                if dataset_name == 'cifar10c' or dataset_name == 'cifar100c':
                    for corruption_type in cifar10c_corruption_types:

                        if check_entry_exists('data.json',method, dataset_name+' '+corruption_type):
                            continue

                        print(f"Processing {dataset_name} with corruption {corruption_type}.")
                        
                        cfg = get_config_file(config_path, dataset_name)
                        print("\nRunning dataset configurations:")
                        print(cfg, "\n")
                        
                        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, corruption_type)
                        clip_weights = clip_classifier(classnames, template, clip_model, preprocess)

                        acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
                        data = {
                        "method": method,
                        "benchmark": dataset_name+' '+corruption_type,
                        "accuracy": round(acc, 4),  # in percentage
                        }
                        write_to_json('data.json', data)
                else:

                    if check_entry_exists('data.json', method, dataset_name):
                            continue
                    
                    # torch.cuda.reset_peak_memory_stats()
                    print(f"Processing {dataset_name} dataset.")
                    
                    cfg = get_config_file(config_path, dataset_name)
                    print("\nRunning dataset configurations:")
                    print(cfg, "\n")
                    
                    test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
                    clip_weights = clip_classifier(classnames, template, clip_model, preprocess)

                    acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
                    data = {
                        "method": method,
                        "benchmark": dataset_name,
                        "accuracy": round(acc, 4),  # in percentage
                        }
                    write_to_json('data.json', data)

            del clip_model
            del clip_weights
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()