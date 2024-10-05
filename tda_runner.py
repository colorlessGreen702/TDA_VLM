import random
import argparse
import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn.functional as F
import operator

import csv
import clip
from utils import *
import json
import time


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

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits
    
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


# def check_entry_exists(file_name, lock, method, benchmark):
#     with lock:
#         try:
#             # Read the existing data from the JSON file
#             with open(file_name, 'r') as file:
#                 existing_data = json.load(file)
#         except FileNotFoundError:
#             # If the file does not exist, return False
#             return False

#         # Check if an entry with the same method and benchmark exists
#         for entry in existing_data:
#             if entry.get('method') == method and entry.get('benchmark') == benchmark:
#                 return True

#         # Return False if no match is found
#         return False

def profile_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, dataset):
    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        measured_samples = 0
        latency = 0
        memory_usage = 0
        
        #Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        # pos_enabled, neg_enabled = False, False
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        #Test-time adaptation
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):

            if i >= 20:
                start_time = time.time()
                torch.cuda.reset_peak_memory_stats('cuda')

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
            
            if i >= 20:
                latency += time.time() - start_time
                memory_usage += torch.cuda.max_memory_allocated('cuda')
                measured_samples += 1
                if measured_samples == 10:
                    break

        avg_memory = memory_usage / measured_samples
        avg_latency = latency / measured_samples

        data = {
            "method": 'TDA',
            "benchmark": dataset,
            "memory": round(avg_memory / (1024 ** 2), 4),  # Convert memory to MB
            "latency": round(avg_latency, 4),  # Latency per sample in seconds
        }

        write_to_json('data.json', data)

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

        print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    cifar10c_corruption_types = ['original','gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 
                                 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 
                                 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 
                                 'jpeg_compression']
    
    # Run TDA on each dataset
    datasets = ['cifar10', 'cifar100', 'caltech101', 'dtd', 'oxford_pets', 'ucf101', 'A', 'V']


    for dataset_name in datasets:
        if dataset_name == 'cifar10' or dataset_name == 'cifar100':
            for corruption_type in cifar10c_corruption_types:
                print(f"Processing {dataset_name} with corruption {corruption_type}.")
                
                cfg = get_config_file(config_path, dataset_name)
                print("\nRunning dataset configurations:")
                print(cfg, "\n")
                
                test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess, corruption_type)
                clip_weights = clip_classifier(classnames, template, clip_model)

                # acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
                # data = {
                #     "method": 'TDA',
                #     "benchmark": f'{dataset_name} {corruption_type}',
                #     "accuracy": acc  # in percentage
                #     }
                # write_to_json('data.json', data)

                profile_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, f'{dataset_name} {corruption_type}')

        else:
            print(f"Processing {dataset_name} dataset.")
            
            cfg = get_config_file(config_path, dataset_name)
            print("\nRunning dataset configurations:")
            print(cfg, "\n")
            
            test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
            clip_weights = clip_classifier(classnames, template, clip_model)

            # acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
            # data = {
            #     "method": 'TDA',
            #     "benchmark": dataset_name,
            #     "accuracy": acc  # in percentage
            #     }
            # write_to_json('data.json', data)

            profile_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights, f'{dataset_name}')



if __name__ == "__main__":
    main()