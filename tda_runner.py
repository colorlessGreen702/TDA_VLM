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
import tent

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
        # pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        pos_enabled, neg_enabled = False, False
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
            wandb.log({"Averaged test accuracy": sum(accuracies)/len(accuracies)}, commit=True)

            if i%1000==0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))
        print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies)/len(accuracies)))   
        return sum(accuracies)/len(accuracies)
    
def append_to_csv(file_path, header, data):
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        if not file_exists:
            writer.writerow(header)
        
        writer.writerow(data)



def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    nbits_list = [8]
    # group_sizes = [128, 64, 32, 16, 8]
    group_sizes = [1]
    header = ['Model', 'caltech101', 'dtd', 'oxford_pets', 'ucf101', 'ImageNetA', 'ImageNetV', 'Average']
    # header = ['Model','ImageNetA', 'ImageNetV', 'Average']
    max_mem_header = header[:1] + ['Initial mem (MB)'] + header[1:]
    time_header = header + ['Total']


    for nb in nbits_list:
        for gp in group_sizes:
            model, acc_data, max_mem, init_mem, runtimes = [], [], [], [], []

            # quant_config = HqqConfig(nbits=nb, group_size=gp, quant_zero=False, quant_scale=False, axis=0)
            # if nb == 8:
            #     quant_config = BitsAndBytesConfig(load_in_8bit=True)
            # elif nb == 4:
            #     quant_config = BitsAndBytesConfig(load_in_4bit=True)

            # quantos = "int" + str(nb)
            # print(quantos)

            # quant_config = QuantoConfig(weights=quantos)

            quant_config = None
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float16, device_map="cuda", quantization_config=quant_config)
            # prepare_for_inference(clip_model)
            preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

            # Tent and ClipArTT TTA
            clip_model = tent.configure_model(clip_model, 'ViT')
            params, param_names = tent.collect_params(clip_model, 'ViT')
            print(param_names)
            optimizer = torch.optim.SGD(params, 0.001/64, momentum=0.9) 
            clip_model = tent.Tent(clip_model, optimizer, method='tent')


            mem = torch.cuda.memory_allocated() / 1024 ** 2
            print(f"Memory allocated after model is on GPU: {mem:.2f} MB")
            init_mem.append(round(mem, 2))

            # Set random seed
            random.seed(1)
            torch.manual_seed(1)

            if args.wandb:
                date = datetime.now().strftime("%b%d_%H-%M-%S")
                group_name = f"{args.backbone}_{args.datasets}_{date}"
            
            # model.append(f'HQQ {nb}bit {gp}gp HF Prompts No TDA')
            # model.append(f'BnB {nb}bit HF Prompts No TDA')

            model.append(f'HF Prompts No TDA')


            # Run TDA on each dataset
            datasets = args.datasets.split('/')
            for dataset_name in datasets:
                torch.cuda.reset_peak_memory_stats()
                clip_model.reset()
                print(f"Processing {dataset_name} dataset.")
                
                cfg = get_config_file(config_path, dataset_name)
                print("\nRunning dataset configurations:")
                print(cfg, "\n")
                
                test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
                clip_weights = clip_classifier(classnames, template, clip_model, preprocess)

                if args.wandb:
                    run_name = f"{dataset_name}"
                    run = wandb.init(project="CLIP-ViT-B16-VM", config=cfg, group=group_name, name=run_name)
                else:
                    run = wandb.init(mode='disabled')

                start = timeit.default_timer()
                acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
                stop = timeit.default_timer()
                time = stop - start
                runtimes.append(round(time, 2))
                acc_data.append(round(acc, 5))
                max_mem.append(round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2))
                if args.wandb:
                    wandb.log({f"{dataset_name}": acc})
                    run.finish()
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
            acc_metrics = model + acc_data + [round(sum(acc_data)/len(acc_data), 5)]
            mem_metrics = model + init_mem + max_mem + [round(sum(max_mem)/len(max_mem), 2)]
            time_metrics = model + runtimes + [round(sum(runtimes)/len(runtimes), 2)] + [round(sum(runtimes), 2)]
            append_to_csv('acc_metrics.csv', header, acc_metrics)
            append_to_csv('mem_metrics.csv', max_mem_header, mem_metrics)
            append_to_csv('time_metrics.csv', time_header, time_metrics)
            del clip_model
            del clip_weights
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()