# MIST

## Training MIST

```bash 
TRAIN_FILE=/your/path/to/train.json
VALID_FILE=/your/path/to/valid.json
OUTPUT_DIR=/your/path/to/save_checkpoints
CACHE_DIR=/your/path/to/transformer_package_cache

MODEL_PATH=bert-base-uncased or models/unilm1.2-base-uncased

# squadqg 30005 steps
# squadqg 50005 steps
# xsum 600005 steps
STEPS=30005

python -m torch.distributed.launch --nproc_per_node=4 train.py\
  --train_file $TRAIN_FILE\
  --valid_file $VALID_FILE\
  --output_dir $OUTPUT_PATH\
  --model_type nat --model_name_or_path $MODEL_PATH\
  --do_lower_case --max_source_seq_length 464 --max_target_seq_length 48\
  --per_gpu_train_batch_size 16 --gradient_accumulation_steps 1\
  --learning_rate 3e-5 --num_warmup_steps 500 --num_training_steps $STEPS\
  --cache_dir $CACHE_DIR\
  --log_dir ${OUTPUT_PATH}/log\
  --keep_prob 0.0\
  --random_prob 0.0\
  --use_glat\
  --tqdm_miniters 100\
  --cotrain_put_target_in_source\ 
  --cotrain_put_target_in_source_same_bert\ 
  --wandb\ # logging with wandb
  --fp16\
  --fp16_opt_level O2
```
Removing the `cotrain_put_target_in_source` and `cotrain_put_target_in_source_same_bert` flags to reproduce the results without MIST.


## Download Unilm
```bash
mkdir -p models/unilm1.2-base-uncased
cd models/unilm1.2-base-uncased
wget https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin -O pytorch_model.bin
wget https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-vocab.txt -O vocab.txt
wget https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased-config.json -O config.json
```

## Download datasets
Json dataset links: [squadqg](https://drive.google.com/file/d/1YR-xohW7UQZNNQG57o34OOM2TiQeY5J8/view?usp=sharing), [xsum](https://drive.google.com/file/d/1fgCs2iEHnU5uGOXprcE6OVtx_PXp806e/view?usp=sharing) and [quora](https://drive.google.com/file/d/112ZLEe3V4guHtZZXuO8f-b_Yrk6_MG7b/view?usp=sharing)


## Training NAT MASS

To reproduce the results of NAT MASS, please refer to the ./MASS-NAT/mass-nat.sh
