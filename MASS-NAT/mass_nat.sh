#!/usr/bin/env bash
set -euo pipefail

cp -r $1 qg
mkdir -p qg_p
wget https://raw.githubusercontent.com/microsoft/MASS/master/MASS-summarization/encode.py
for SPLIT in train dev test; do
  for LANG in tgt src
  do
    python encode.py \
        --inputs qg/${SPLIT}.$LANG \
        --outputs qg_p/${SPLIT}.$LANG\
        --workers 60; \
 done
done

for SPLIT in train dev test; do
    sed -i 's/\[ \[UNK\] \]/[SEP]/' qg_p/${SPLIT}.src
    sed -i 's/\[ \[UNK\] \]/[UNK]/g' qg_p/${SPLIT}.src
done

wget -c https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz
tar -zxvf mass-base-uncased.tar.gz

cp /mnt/phily/mass.zip .
unzip mass.zip

fairseq-preprocess \
    --user-dir  mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref qg_p/train --validpref qg_p/dev --testpref qg_p/test \
    --destdir data --srcdict dict.txt --tgtdict dict.txt \
    --workers 20

CHECKPOINT=checkpoints_mass_nat_glat
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train data \
      --user-dir  mass --task translation_nat_mass --arch transformer_nat_mass_base \
      --criterion nat_loss\
      --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
      --lr 3e-5 --min-lr 1e-09\
      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 500 \
      --weight-decay 0.0 \
      --label-smoothing 0.1 \
      --update-freq 1 --max-sentences 16\
      --ddp-backend=no_c10d --max-epoch 50 \
      --max-source-positions 512 --max-target-positions 512 \
      --skip-invalid-size-inputs-valid-test \
      --save-dir ${CHECKPOINT}\
      --keep-last-epochs 1\
       --load-from-pretrained-model mass-base-uncased.pt\
       --fp16  --glat

cp -r ${CHECKPOINT} /mnt/phily/fairseq/

#sudo sed -i 's/torch.div/torch.floor_divide/' /opt/conda/lib/python3.7/site-packages/fairseq/search.py

CHECKPOINT=checkpoints_mass_nat_glat
MODEL=/mnt/phily/fairseq/${CHECKPOINT}/checkpoint_best.pt
DATADIR=data
TEMP_FILE=fairseq_outputs.txt
OUTPUT_FILE=sorted_outputs.txt
MODEL=/mnt/phily/fairseq/${CHECKPOINT}/checkpoint_best.pt
CUDA_VISIBLE_DEVICES=1 fairseq-generate $DATADIR --path $MODEL \
    --user-dir MASS/MASS-summarization/mass --task translation_nat_mass \
    --gen-subset test\
    --batch-size 64 --beam 6 --min-len 6 --no-repeat-ngram-size 3 \
    --lenpen 1.0 > $TEMP_FILE
#cp $TEMP_FILE  /mnt/phily/fairseq/${CHECKPOINT}
