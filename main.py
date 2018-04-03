import json, math, os
import argparse
from tqdm import tqdm
import numpy as np
import tensorflow as tf 
from collections import Counter
from pprint import pprint
from model import Model, get_model
from graph_handler import GraphHandler
from evaluate import Evaluator
from load_data import read_reuters
from load_newsgroup import read_news
from common import DataSet

def main(config):
  with tf.device("/gpu:0"):
    if config.mode == "train":
      _train(config)
    elif config.mode == "test":
      _test(config)
    elif config.mode == "forward":
      _forward(config)
    elif config.mode == "check":
      _check(config)
    else:
      raise ValueError("invalid value for 'mode': {}".format(config.mode))

def _train(config):
  word2idx = Counter(json.load(open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "r"))["word2idx"])
  idx2word = json.load(open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "r"))["idx2word"]
  assert len(word2idx) == len(idx2word)
  for i in range(10):  assert word2idx[idx2word[i]] == i 
  vocab_size = len(word2idx)
  print("vocab_size", vocab_size, idx2word[:10])
  word2vec = Counter(json.load(open("../data/{}/word2vec_{}.json".format(config.data_from, config.pretrain_from), "r"))["word2vec"])
  # word2vec = {} if config.debug or config.load  else get_word2vec(config, word2idx)
  idx2vec = {word2idx[word]: vec for word, vec in word2vec.items() if word in word2idx}
  print("no unk words:", len(idx2vec))

  unk_embedding = np.random.multivariate_normal(np.zeros(config.word_embedding_size), np.eye(config.word_embedding_size))
  config.emb_mat = np.array([idx2vec[idx] if idx in idx2vec else unk_embedding for idx in range(vocab_size)])
  config.vocab_size = vocab_size 
  print("emb_mat:", config.emb_mat.shape)
  
  train_dict, test_dict = {}, {}
  if os.path.exists("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "train")):  
    train_dict = json.load(open("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "train"), "r"))
  if os.path.exists("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "test")):  
    test_dict = json.load(open("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "test"), "r"))

  # check 
  for key, val in train_dict.items():
    if isinstance(val[0], list) and len(val[0])>10: print(key, val[0:4])
    else: print(key, val[0:4])

  if config.data_from == "reuters":
    train_data = DataSet(train_dict, "train") if len(train_dict)>0 else read_reuters(config, data_type="train", word2idx=word2idx)
    dev_data = DataSet(test_dict, "test") if len(test_dict)>0 else read_reuters(config, data_type="test", word2idx=word2idx)
  elif config.data_from == "20newsgroup":
    train_data = DataSet(train_dict, "train") if len(train_dict)>0 else read_news(config, data_type="train", word2idx=word2idx)
    dev_data = DataSet(test_dict, "test") if len(test_dict)>0 else read_news(config, data_type="test", word2idx=word2idx)

  config.train_size = train_data.get_data_size()
  config.dev_size = dev_data.get_data_size()
  print("train/dev:", config.train_size, config.dev_size)
  if config.max_docs_length > 2000:  config.max_docs_length = 2000 
  pprint(config.__flags, indent=2)
  model = get_model(config)
  graph_handler = GraphHandler(config, model)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  graph_handler.initialize(sess)

  num_batches = config.num_batches or int(math.ceil(train_data.num_examples / config.batch_size)) * config.num_epochs
  global_step = 0

  dev_evaluate = Evaluator(config, model) 

  for batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_batches, shuffle=True, cluster=config.cluster), total=num_batches):
    batch_idx, batch_ds = batch
    '''
    if config.debug:
      for key, value in batch_ds.data.items():
        if not key.startswith("x"):
          print(key, value)
      continue
    '''
    global_step = sess.run(model.global_step) + 1
    # print("global_step:", global_step)
    get_summary = global_step % config.log_period  
    feed_dict = model.get_feed_dict(batch, config)
    xx_context, xx_final = sess.run([model.xx_context, model.xx_final], feed_dict=feed_dict)
    print("check: ", xx_context.shape, xx_context, xx_final.shape, xx_final)
    logits, y, y_len, loss, summary, train_op = sess.run([model.logits, model.y, model.y_seq_length,  model.loss, model.summary, model.train_op], feed_dict=feed_dict)
    print("logits:", logits[0:3], y[0:3], y_len[0:3], logits.shape, y.shape, y_len.shape) 
    print("loss:", loss)
    if get_summary:
      graph_handler.add_summary(summary, global_step)
    # occasional saving
    if global_step % config.save_period == 0:
      graph_handler.save(sess, global_step=global_step)
    if not config.eval:
      continue
    # Occasional evaluation
    if global_step % config.eval_period == 0:
      #config.test_batch_size = config.dev_size/3
      num_steps = math.ceil(dev_data.num_examples / config.test_batch_size)
      if 0 < config.val_num_batches < num_steps:
        num_steps = config.val_num_batches
      # print("num_steps:", num_steps)
      e_dev = dev_evaluate.get_evaluation_from_batches(
        sess, tqdm(dev_data.get_batches(config.test_batch_size, num_batches=num_steps), total=num_steps))
      graph_handler.add_summaries(e_dev.summaries, global_step)
      
def _check(config):
  word2idx = Counter(json.load(open("../data/word2idx_new.json", "r"))["word2idx"])
  vocab_size = len(word2idx)
  #word2vec = {} # or get_word2vec(word2idx)
  word2vec = Counter(json.load(open("../data/word2vec_{}.json".format(config.pretrain_from), "r"))["word2vec"])
  idx2vec = {word2idx[word]: vec for word, vec in word2vec.items() if word in word2idx and word!="UNK"}
  unk_embedding = np.random.multivariate_normal(np.zeros(config.word_embedding_size), np.eye(config.word_embedding_size))
  config.emb_mat = np.array([idx2vec[idx] if idx in idx2vec else unk_embedding for idx in range(vocab_size)])
  config.vocab_size = vocab_size 
  print("emb_mat:", config.emb_mat.shape)

  if config.data_from == "reuters":
    train_data = read_reuters(config, data_type="train", word2idx=word2idx)
    dev_data = read_reuters(config, data_type="test", word2idx=word2idx)
  elif config.data_from == "20newsgroup":
    train_data = read_news(config, data_type="train", word2idx=word2idx)
    dev_data = read_newss(config, data_type="test", word2idx=word2idx)
  config.train_size = train_data.get_data_size()
  config.dev_size = dev_data.get_data_size()
  pprint(config.__flags, indent=2)
  model = get_model(config)
  graph_handler = GraphHandler(config, model)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  graph_handler.initialize(sess)

  num_batches = config.num_batches or int(math.ceil(train_data.num_examples / config.batch_size)) * config.num_epochs
  global_step = 0

  for batch in tqdm(train_data.get_batches(config.batch_size, num_batches=num_batches, shuffle=True, cluster=config.cluster), total=num_batches):
    batch_idx, batch_ds = batch
    global_step = sess.run(model.global_step) + 1
    # print("global_step:", global_step)
    get_summary = global_step % config.log_period  
    feed_dict = model.get_feed_dict(batch, config)
    check, xx_final, xx_context =  sess.run([model.check, model.xx_final, model.xx_context], feed_dict=feed_dict)
    print("check:", check.shape, type(check), xx_final.shape, xx_context.shape)

def _test(config):
  word2idx = Counter(json.load(open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "r"))["word2idx"])
  idx2word = json.load(open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "r"))["idx2word"]
  assert len(word2idx) == len(idx2word)
  for i in range(10):  assert word2idx[idx2word[i]] == i
  vocab_size = len(word2idx)
  word2vec = Counter(json.load(open("../data/{}/word2vec_{}.json".format(config.data_from, config.pretrain_from), "r"))["word2vec"])
  # word2vec = {} if config.debug or config.load  else get_word2vec(config, word2idx)
  idx2vec = {word2idx[word]: vec for word, vec in word2vec.items() if word in word2idx}
  unk_embedding = np.random.multivariate_normal(np.zeros(config.word_embedding_size), np.eye(config.word_embedding_size))
  config.emb_mat = np.array([idx2vec[idx] if idx in idx2vec else unk_embedding for idx in range(vocab_size)])
  config.vocab_size = vocab_size 
  train_dict, test_dict = {}, {}
  if os.path.exists("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "test")):  
    test_dict = json.load(open("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "test"), "r"))

  if config.data_from == "reuters":
    dev_data = DataSet(test_dict, "test") if len(test_dict)>0 else read_reuters(config, data_type="test", word2idx=word2idx)
  elif config.data_from == "20newsgroup":
    dev_data = DataSet(test_dict, "test") if len(test_dict)>0 else read_news(config, data_type="test", word2idx=word2idx)

  config.dev_size = dev_data.get_data_size()
  # if config.use_glove_for_unk:
  pprint(config.__flags, indent=2)
  model = get_model(config)
  graph_handler = GraphHandler(config, model)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  graph_handler.initialize(sess)
  # check 
  #w_embeddings = sess.run(model.word_embeddings)
  #print("w_embeddings:", w_embeddings.shape, w_embeddings)
    
  dev_evaluate = Evaluator(config, model) 
  num_steps = math.ceil(dev_data.num_examples / config.test_batch_size)
  if 0 < config.val_num_batches < num_steps:
    num_steps = config.val_num_batches
  # print("num_steps:", num_steps)
  e_dev = dev_evaluate.get_evaluation_from_batches(
    sess, tqdm(dev_data.get_batches(config.test_batch_size, num_batches=num_steps), total=num_steps))

def _forward(config):
  pass
