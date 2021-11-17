#!/usr/bin/env python3
import os
import json
import tqdm
import torch
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def split_json(json_file):
    src_data = open(json_file.replace('.json', '.src'), 'w')
    tgt_data = open(json_file.replace('.json', '.tgt'), 'w')

    with open(json_file) as f:
        for l in f:
            t = json.loads(l)
            src_data.write(t['src'] + '\n')
            tgt_data.write(t['tgt'] + '\n')
    src_data.close()
    tgt_data.close()

def merge_to_json(path, src='.src', tgt='.tgt'):
    src_data = open(path+src).readlines()
    tgt_data = open(path+tgt).readlines()

    s_lens, t_lens = [], []
    with open(path+'.json', 'w') as f:
        for s, t in zip(src_data, tgt_data):
            s, t = s.replace('\n', ''), t.replace('\n', '')
            s_lens.append(len(s.split()))
            t_lens.append(len(t.split()))
            f.write(json.dumps({'src': s, "tgt": t})+'\n')
    logging.info(f'src avg len: {np.mean(s_lens)} min {min(s_lens)} max {max(s_lens)}')
    logging.info(f'tgt avg len: {np.mean(t_lens)} min {min(t_lens)} max {max(t_lens)}')


def merge_to_kd_json(path, kd_type='nat'):
    src_data = open(path+'.src').readlines()
    tgt_data = open(path+'.tgt').readlines()
    tgt_kd_data = open(path+f'_{kd_type}_kd.tgt').readlines()

    s_lens, t_lens = [], []
    with open(path+f'_{kd_type}_kd.json', 'w') as f:
        for s, t, k in zip(src_data, tgt_data, tgt_kd_data):
            s, t, k = s.replace('\n', ''), t.replace('\n', ''), k.replace('\n', '')
            s_lens.append(len(s.split()))
            t_lens.append(len(t.split()))
            f.write(json.dumps({'src': s, "tgt": t, "tgt_kd": k})+'\n')

    s_lens, t_lens = [], []
    with open(path+f'_{kd_type}_kd_only.json', 'w') as f:
        for s, t in zip(src_data, tgt_kd_data):
            s, t = s.replace('\n', ''), t.replace('\n', '')
            s_lens.append(len(s.split()))
            t_lens.append(len(t.split()))
            f.write(json.dumps({'src': s, "tgt": t})+'\n')

    logging.info(f'src avg len: {np.mean(s_lens)} min {min(s_lens)} max {max(s_lens)}')
    logging.info(f'tgt avg len: {np.mean(t_lens)} min {min(t_lens)} max {max(t_lens)}')

def load_and_cache_examples_xsum(
        mode, tokenizer, local_rank, cached_features_file, shuffle=True, kd_file=None):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()


    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        import datasets
        xsum = datasets.load_dataset('xsum', cache_dir='./data')[mode]
        examples = []
        features = []

        if kd_file is not None:
            with open(kd_file) as f:
                kd_summary = [i.replace('\n', '') for i in f]

        for i in tqdm.tqdm(range(len(xsum))):
            source_tokens = tokenizer.tokenize(xsum[i]["document"])
            if kd_file is None:
                target_tokens = tokenizer.tokenize(xsum[i]["summary"])
            else:
                target_tokens = tokenizer.tokenize(kd_summary[i])
            features.append({
                    "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                    "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
                })

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def load_and_cache_examples(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))
        features = []

        for example in tqdm.tqdm(examples):
            if isinstance(example["src"], list):
                source_tokens = example["src"]
                target_tokens = example["tgt"]
            else:
                source_tokens = tokenizer.tokenize(example["src"])
                target_tokens = tokenizer.tokenize(example["tgt"])
            features.append({
                    "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                    "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
                })

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def load_and_cache_examples_two_stage(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))
        features = []

        for example in tqdm.tqdm(examples):
            if isinstance(example["src"], list):
                source_tokens = example["src"]
                target_kd_tokens = example["tgt_kd"]
                target_tokens = example["tgt"]
            else:
                source_tokens = tokenizer.tokenize(example["src"])
                target_kd_tokens = tokenizer.tokenize(example["tgt_kd"])
                target_tokens = tokenizer.tokenize(example["tgt"])
            features.append({
                    "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
                    "target_kd_ids": tokenizer.convert_tokens_to_ids(target_kd_tokens),
                    "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
                })

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def _fast_process_one(example, tokenizer):
    if isinstance(example["src"], list):
        source_tokens = example["src"]
        target_tokens = example["tgt"]
        if "tgt_kd" in example:
            target_kd_tokens = example["tgt_kd"]
    else:
        source_tokens = tokenizer.tokenize(example["src"])
        target_tokens = tokenizer.tokenize(example["tgt"])
        if "tgt_kd" in example:
            target_kd_tokens = tokenizer.tokenize(example["tgt_kd"])
    out_dict = {
            "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
            "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
        }
    if "tgt_kd" in example:
        out_dict["target_kd_ids"] = tokenizer.convert_tokens_to_ids(target_kd_tokens)
    return out_dict

def load_and_cache_examples_fast(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=True):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()


    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        with open(example_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                examples.append(json.loads(line))

        logger.info("Staring processing")
        import time; b = time.time()
        from multiprocessing import Pool
        with Pool() as p:
            features = p.starmap(_fast_process_one, [(i, tokenizer) for i in examples])
        b = time.time() - b 
        logger.info("End took %s s", b)

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features


def _fast_process_wmt(example, tokenizer):
    example = [e.replace('@@ ', '').replace('@-@', '-') for e in example]
    example = [e.replace(' &quot;', '"').replace(' &apos;', '\'') for e in example]
    src, tgt = example[0], example[1]
    source_tokens = tokenizer.tokenize(src)
    target_tokens = tokenizer.tokenize(tgt)
    out_dict = {
            "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
            "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
        }
    return out_dict

def load_and_cache_examples_wmt(
        source_file, target_file, tokenizer, local_rank, cached_features_file, shuffle=False):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()


    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s %s", source_file, target_file)

        src_examples = open(source_file, mode="r", encoding="utf-8")
        tgt_examples = open(target_file, mode="r", encoding="utf-8")
        examples = zip(src_examples, tgt_examples)
        logger.info("Staring processing")
        import time; b = time.time()
        from multiprocessing import Pool
        with Pool() as p:
            features = p.starmap(_fast_process_wmt, [(i, tokenizer) for i in examples])
        b = time.time() - b
        logger.info("End took %s s", b)
        src_examples.close()
        tgt_examples.close()

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

def _fast_process_ads(example, tokenizer):
    src, tgt = example.replace('\n', '').split('\t')
    source_tokens = tokenizer.tokenize(src)
    target_tokens = tokenizer.tokenize(tgt)
    out_dict = {
            "source_ids": tokenizer.convert_tokens_to_ids(source_tokens),
            "target_ids": tokenizer.convert_tokens_to_ids(target_tokens),
        }
    return out_dict

def load_and_cache_examples_ads(
        example_file, tokenizer, local_rank, cached_features_file, shuffle=False):
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()


    if cached_features_file is not None and os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", example_file)

        examples = []
        examples =  open(example_file, mode="r", encoding="utf-8")
        logger.info("Staring processing")
        import time; b = time.time()
        from multiprocessing import Pool
        with Pool() as p:
            features = p.starmap(_fast_process_ads, [(i, tokenizer) for i in examples])
        b = time.time() - b
        logger.info("End took %s s", b)
        examples.close()

        if shuffle:
            random.shuffle(features)

        if local_rank in [-1, 0] and cached_features_file is not None:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if local_rank == 0:
        torch.distributed.barrier()

    return features

if __name__ == '__main__':
    logging.basicConfig(level = logging.INFO)
    data_path = './data/squadqg_data/org_data'
    merge_to_json(data_path+'/train_ar_kd_last')
    merge_to_json(data_path+'/train_ar_kd_best')
    #merge_to_json(data_path+'/train', src='.document', tgt='.summary')
    #merge_to_json(data_path+'/test', src='.document', tgt='.summary')
    #merge_to_json(data_path+'/validation', src='.document', tgt='.summary')
    #merge_to_kd_json(data_path+'/train',kd_type='6iters')
