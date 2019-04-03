#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

export BERT_BASE_DIR=/tcxia/bert-tian/multi_cased_L-12_H-768_A-12
export DATA_DIR=/tcxia/bert-tian/dataset

python run_classifier.py \
  --task_name=MYSELF \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --do_lower_case=False \
  --output_dir=/tmp/my_output/