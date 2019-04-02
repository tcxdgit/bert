import tensorflow as tf
import os
flags = tf.flags
FLAGS = flags.FLAGS

# export BERT_BASE_DIR=/tcxia/bert-tian/multi_cased_L-12_H-768_A-12
# export DATA_DIR=/tcxia/bert-tian/dataset
# #export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier
# export TRAINED_CLASSIFIER=/tmp/myself_output/
#
# python run_classifier.py \
#   --task_name=MYSELF \
#   --do_predict=true \
#   --data_dir=$DATA_DIR \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#   --init_checkpoint=$TRAINED_CLASSIFIER \
#   --max_seq_length=128 \
#   --output_dir=/tmp/myself_output/

BERT_BASE_DIR = "/tcxia/bert-tian/multi_cased_L-12_H-768_A-12"

# task_name = 'MYSELF'
# do_predict = True
# data_dir = None
# vocab_file = os.path.join(BERT_BASE_DIR, "vocab.txt")
# bert_config_file = os.path.join(BERT_BASE_DIR, "bert_config.json")
# init_checkpoint = "/tmp/myself_output/"
# max_seq_length = 128
# output_dir = "/tmp/myself_output/"

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", os.path.join(BERT_BASE_DIR, "bert_config.json"),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'MYSELF', "The name of the task to train.")

flags.DEFINE_string("vocab_file", os.path.join(BERT_BASE_DIR, "vocab.txt"),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", "/tcxia/myself_output/",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")