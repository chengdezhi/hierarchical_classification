from tqdm import tqdm
from nltk.corpus import stopwords, reuters
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re, json, os, math, random, itertools, numpy
from itertools import zip_longest
from collections import Counter, defaultdict

class MyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, numpy.integer):
      return int(obj)
    elif isinstance(obj, numpy.floating):
      return float(obj)
    elif isinstance(obj, numpy.ndarray):
      return obj.tolist()
    else:
      return super(MyEncoder, self).default(obj)

def gen_word2vec(word_counter, pretrain_from = "wiki.en.vec", data_from = "ice"):
  word2vec_dict = {}
  if pretrain_from == "wiki.en.vec":
    w2v_path = "/data/dechen/w2v/wiki.en.vec"
    w2v_f = open(w2v_path, "r")
  else:
    w2v_path = os.path.join("/home/t-dechen/data/glove", "glove.{}.{}d.txt".format("6B", 300))
    w2v_f = open(w2v_path, "r")
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes['6B']
  w2v = [line for line in w2v_f]
  total = len(w2v)

  for line in tqdm(w2v, total=total):
    array = line.lstrip().rstrip().split(" ")
    word = array[0]
    vector = list(map(float, array[1:]))
    if word in word_counter:
      word2vec_dict[word] = vector
    elif word.capitalize() in word_counter:
      word2vec_dict[word.capitalize()] = vector
    elif word.lower() in word_counter:
      word2vec_dict[word.lower()] = vector
    elif word.upper() in word_counter:
      word2vec_dict[word.upper()] = vector
  shared = {"word2vec": word2vec_dict}
  json.dump(shared, open("../data/{}/word2vec_{}.json".format(data_from, pretrain_from), "w"))
  print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), w2v_path))

def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
  args = [iter(iterable)] * n
  out = zip_longest(*args, fillvalue=fillvalue)
  out = list(out)
  if num_groups is not None:
    default = (fillvalue, ) * n
    assert isinstance(num_groups, int)
    out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
  if shorten:
    assert fillvalue is None
    out = (tuple(e for e in each if e is not None) for each in out)
  return out

class DataSet(object):
  def __init__(self, data, data_type, valid_idxs= None):
    self.data = data  # {x:[], y:[]}  etc
    self.data_type = data_type
    total_num_examples = self.get_data_size()
    self.valid_idxs = range(total_num_examples) if valid_idxs is None else valid_idxs
    self.num_examples = len(self.valid_idxs)

  def _sort_key(self, idx):
    x = self.data["x"][idx]
    return len(x)

  def get_data_size(self):
    assert isinstance(self.data, dict)
    return len(next(iter(self.data.values())))

  def get_by_idxs(self, idxs):
    assert isinstance(self.data, dict)
    out = defaultdict(list)
    for key, val in self.data.items():
      out[key].extend(val[idx] for idx in idxs)
    return out

  def get_batches(self, batch_size, num_batches=None, shuffle=False, cluster=False):
    num_batches_per_epoch = int(math.ceil(self.num_examples / batch_size))
    if num_batches is None:
      num_batches = num_batches_per_epoch
    num_epochs = int(math.ceil(num_batches / num_batches_per_epoch))

    if shuffle:
      random_idxs = random.sample(self.valid_idxs, (num_batches_per_epoch-1)*batch_size) # shuffle
      if cluster:
        sorted_idxs = sorted(random_idxs, key=self._sort_key)
        sorted_grouped = lambda: list(grouper(sorted_idxs, batch_size))
        grouped = lambda: random.sample(sorted_grouped(), num_batches_per_epoch)
      else:
        random_grouped = lambda: list(grouper(random_idxs, batch_size))
        grouped = random_grouped
    else:
      raw_grouped = lambda: list(grouper(self.valid_idxs, batch_size))
      grouped = raw_grouped

    batch_idx_tuples = itertools.chain.from_iterable(grouped() for _ in range(num_epochs))

    for _ in range(num_batches):
      batch_idxs = tuple(i for i in next(batch_idx_tuples) if i is not None)
      batch_data = self.get_by_idxs(batch_idxs)
      batch_ds = DataSet(batch_data, self.data_type)
      print("batch_idxs:", len(batch_idxs))
      yield batch_idxs, batch_ds

def prediction_with_threshold(config, t_preds, t_scores, threshold):
  if config.model_name.endswith("flat"):
    new_preds = []
    for i in range(t_preds.shape[0]):
      single = []
      max_score, max_index = -10.0, 0
      for j in range(t_preds.shape[1]):
        if config.data_from == "20newsgroup" or config.data_from == "ice":
          if t_scores[i,j] > max_score:
            max_score = t_scores[i,j]
            max_index = j
        else:
          if t_scores[i, j] > threshold:
            single += [t_preds[i,j]]
      if config.data_from == "20newsgroup": single += [max_index]
      if len(single)==0:
        single += [max_index]
      new_preds.append(single)
    return new_preds
  else:   # sorted
    t_preds[t_preds == -1] = 0  # set -1 to 0
    new_preds = []
    t_preds = t_preds.transpose((0, 2, 1))
    for i in range(t_preds.shape[0]):
      single = []
      for j in range(t_preds.shape[1]):
        if config.data_from == "20newsgroup" and j==0:
          single += t_preds[i,j].tolist()
          break
        if t_scores[i, j] > threshold or j==0:
          single += t_preds[i, j].tolist()
      new_preds.append(single)
    return new_preds
