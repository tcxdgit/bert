python extract_features.py\
 --input_file=input.txt\
 --output_file=predict/output.json\
 --vocab_file=./chinese_L-12_H-768_A-12/vocab.txt\
 --bert_config_file=./chinese_L-12_H-768_A-12/bert_config.json\
 --init_checkpoint=./chinese_L-12_H-768_A-12/bert_model.ckpt\
 --layers=-1,-2,-3,-4\
 --max_seq_length=128\
 --batch_size=8

