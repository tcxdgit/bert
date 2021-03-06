python run_pretraining.py\
  --input_file=./output_example_cn/tf_examples.tfrecord \
  --output_dir=./pre_training_cn\
  --do_train=True\
  --do_eval=True\
  --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json\
  --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt\
  --train_batch_size=10\
  --max_seq_length=300\
  --max_predictions_per_seq=45\
  --num_train_steps=100\
  --num_warmup_steps=10\
  --learning_rate=2e-5\
