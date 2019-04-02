#!/usr/bin/env bash

export BERT_BASE_DIR=/tcxia/bert-tian/multi_cased_L-12_H-768_A-12
export DATA_DIR=/tcxia/bert-tian/dataset
#export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier
export TRAINED_CLASSIFIER=/tmp/myself_output/

python run_classifier.py \
  --task_name=MYSELF \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/myself_output/
