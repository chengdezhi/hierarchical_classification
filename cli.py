import os
import numpy as np
import tensorflow as tf
from main import main as m

flags = tf.app.flags
# device 
flags.DEFINE_string("gpu_ids", "0", "Run ID [0]")
flags.DEFINE_string("device_type", "gpu", "Run ID [0]")
flags.DEFINE_integer("gpu_idx", 0, "")

# data 
flags.DEFINE_string("data_from", "20newsgroup", "data_from")

# training
flags.DEFINE_float("learning_rate", 0.0005, "learning_rate")
flags.DEFINE_float("keep_prob", 0.8, "keep_prob")
flags.DEFINE_integer("num_batches", 600, "")
flags.DEFINE_integer("batch_size", 100, "")
flags.DEFINE_integer("test_batch_size", 133, "")
# TODO check epoch 
flags.DEFINE_integer("num_epochs", 200, "")
flags.DEFINE_integer("log_period", 30, "")
flags.DEFINE_integer("eval_period", 4, "")
flags.DEFINE_integer("save_period", 4, "")
flags.DEFINE_integer("val_num_batches", 0, "")

# network 
flags.DEFINE_integer("word_embedding_size", 300, "")
flags.DEFINE_integer("label_embedding_size", 300, "")
flags.DEFINE_integer("hidden_size", 150, "")
flags.DEFINE_integer("beam_width", 5, "")
flags.DEFINE_float("thred", -2.0, "")
flags.DEFINE_integer("EOS", 20, "")
flags.DEFINE_integer("PAD", 0, "")
flags.DEFINE_integer("GO", 1, "")
flags.DEFINE_boolean("project", False, "")
flags.DEFINE_boolean("concat_w2v", False, "")
flags.DEFINE_boolean("div", False, "")

# graph control
flags.DEFINE_string("mode", "train", "")
flags.DEFINE_string("model_name", "RCNN", "")   # RCNN 
flags.DEFINE_string("load_path", "", "")   # RCNN 
flags.DEFINE_string("pretrain_from", "wiki.en.vec", "")   # RCNN 
flags.DEFINE_integer("max_to_keep", 100, "")

flags.DEFINE_boolean("load", False, "load saved data? [True]")
flags.DEFINE_boolean("load_ema", False, "load saved data? [True]")
flags.DEFINE_integer("load_step", 0, "")  
flags.DEFINE_boolean("eval", True, "eval data? [True]")
flags.DEFINE_boolean("eval_trees", True, "eval trees? [True]")
flags.DEFINE_boolean("eval_layers", True, "eval layers? [True]")
flags.DEFINE_boolean("cluster", False, "cluster data? [False]")
flags.DEFINE_boolean("debug", False, "debug")
flags.DEFINE_boolean("check", False, "check")

# define hierarchical 
flags.DEFINE_integer("max_seq_length", 4, "")
flags.DEFINE_integer("n_classes", 21, "")
flags.DEFINE_integer("max_docs_length", 0, "")

# dir
flags.DEFINE_string("out_dir", "out", "")
# flags.DEFINE_string("save_dir", "out/save", "")
# flags.DEFINE_string("log_dir", "out/log", "")

config = flags.FLAGS

def main(_):
  if config.debug:
    #config.mode = "check"
    config.num_batches = 100
    config.log_period = 1
    config.save_period = 1
    config.eval_period = 1
    config.batch_size = 2
    config.val_num_batches = 3
    config.out_dir = "debug"
  #print(config.test_batch_size)
  if config.model_name.endswith("flat"):  
    if config.data_from=="reuters": config.n_classes = 18
    if config.data_from=="20newsgroup": config.n_classes = 20
    config.thred = 0.053
  else:
    if config.data_from=="reuters": config.n_classes = 21
    if config.data_from=="20newsgroup": config.n_classes = 29
    
  if config.data_from == "reuters":
    config.max_docs_length = 818
    config.tree1 = np.array([2,3,4,5,6,7,8])
    config.tree2 = np.array([9,10,11,12,13,14,15])
    config.tree3 = np.array([16,17,18,19])
    config.layer1 = np.array([2, 9, 16])
    config.layer2 = np.array([3, 4, 10, 11, 17, 19])
    config.layer3 = np.array([5, 6, 7, 8, 12, 13, 14, 15, 18])
  
  if config.data_from == "20newsgroup":
    config.EOS = 28
    config.test_batch_size = 26
    config.max_docs_length = 1000
    config.max_seq_length = 3
    config.tree1 = np.array([22,3,4,5,6,7])
    config.tree2 = np.array([23,9,10,11,12])
    config.tree3 = np.array([24,13,14,15,16])
    config.tree4 = np.array([25,8])
    config.tree5 = np.array([26,10,18,19])
    config.tree6 = np.array([27,21,2,17])
    config.layer1 = np.array([22,23,24,25,26,27])
    config.layer2 = np.array([3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21])


  config.out_dir = os.path.join("../data", config.out_dir)
  config.save_dir = os.path.join(config.out_dir, "save")
  config.log_dir = os.path.join(config.out_dir, "log")
  if not os.path.exists(config.out_dir):  # or os.path.isfile(config.out_dir):
    os.makedirs(config.out_dir)
  if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)
  if not os.path.exists(config.log_dir):
    os.mkdir(config.log_dir)
  os.environ["CUDA_VISIBLE_DEVICES"]=config.gpu_ids
  m(config)

if __name__=="__main__":
  tf.app.run()
