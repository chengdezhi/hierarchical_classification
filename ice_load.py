#!/usr/bin/env python
# coding: utf-8
import csv, json, os, math, re
import numpy as np
from common import MyEncoder, DataSet
from collections import defaultdict, deque
from collections import Counter
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from common import gen_word2vec

def tokenize(text):
  min_length = 2
  tokens = map(lambda word: word.lower(), word_tokenize(text))
  # cachedStopWords = stopwords.words("english")
  # tokens = [word for word in tokens if word not in cachedStopWords]
  '''
  new_tokens = []
  for token in tokens:
    for _ in re.split('\W', token): new_tokens += [_]
  tokens = new_tokens
  '''
  tokens = (list(map(lambda token: PorterStemmer().stem(token), tokens)))
  p = re.compile('[a-zA-Z]+');
  filtered_tokens = list(filter (lambda token: p.match(token) and len(token) >= min_length,tokens))
  return filtered_tokens

def statistic(th):
  # check th layer bfs
  d = deque()
  d.append("root")
  print("root")
  cnt = 0
  showup = set()
  while d:
    cnt += 1
    size = len(d)
    curs = []
    for i in range(size):
      cur = d.popleft()
      showup.add(cur)
      for item in th[cur]:
        if item not in showup:
          d.append(item)
      curs += [cur]
    if cnt==2:
      print(curs)
    print(len(curs))
  print("done!", cnt)

def get_hdict():
  iceid2topic = "/home/t-dechen/data/ICE/ICEId2Topic.tsv"
  lines = [line.split("/") for line in open(iceid2topic, "r")]
  th = defaultdict(set)
  for k,line in enumerate(lines):
    # print("line", line)
    # th["root"].add(line[1].strip())
    line[0] = "root"
    line = [label.strip() for label in line]
    #if k==3: print(line)
    for i in range(len(line)-1):
      par = ",".join(line[0:i+1])
      cld = ",".join(line[0:i+2])
      th[par].add(cld)
  return th

def get_idxs(cand=True):
  if os.path.exists("../data/{}/{}_{}.json".format("ice","ice","label")):
    idx = json.load(open("../data/{}/{}_{}.json".format("ice","ice","label"), "r"))
    id2seqs, topic2id, id2topic = idx["id2seqs"], idx["topic2id"], idx["id2topic"]
    print(id2topic[:10])
    label_id = str(13824)
    print(id2seqs[label_id], label_id)
    return id2seqs, topic2id, id2topic
  iceid2topic = "/home/t-dechen/data/ICE/ICEId2Topic.tsv"
  lines = [line.split("/") for line in open(iceid2topic, "r")]
  th = defaultdict(list)

  id2seqs = defaultdict(list)
  topic2id = defaultdict(int)
  id2topic = []
  id2topic = ["padding", "root"]
  topic2id["padding"] = 0
  topic2id["root"] = 1

  label_ids = set()
  if cand:
    cand_ids = get_candids()
    cand_topics = ["computers consumer electronics", "arts entertainment", "finance"]
  print(cand_ids[:10])

  for k, line in enumerate(lines):
    line = [label.strip() for label in line]
    label_id = int(line[0])
    top = line[1].strip()
    if cand and (top not in cand_topics):
      continue
    assert top in cand_topics
    #print("top:", top, line)
    label_ids.add(label_id)
    seq = [1]
    for topic in line[1:]:
      if topic not in id2topic:
        topic2id[topic] = len(id2topic)
        id2topic += [topic]
      seq += [topic2id[topic]]
      s_size = len(seq)
    for i in range(9-s_size):  seq += [0]
    # pad
    id2seqs[label_id] += [seq]

  topic2id["end"] = len(id2topic)
  print(topic2id["end"])
  if cand: label_ids = cand_ids
  for label_id in label_ids:
    label_id = int(label_id)
    r_seqs = []
    for seq in id2seqs[label_id]:
      cnt = 0
      new_seq = list(seq)
#      print("new_seq:0", new_seq)
      for topic_id in seq:
        if topic_id !=0 : cnt+=1
      if cnt < 9:
        new_seq[cnt] = topic2id["end"]
      r_seqs += [new_seq]
 #     print("new_seq:1", new_seq)
    id2seqs[label_id] = list(r_seqs)

  idx = {
         "id2seqs":id2seqs,
         "topic2id":topic2id,
         "id2topic":id2topic
         }
  #print("idx:", idx)
  print(id2topic[:10])
  print(id2seqs[label_id], label_id)

  json.dump(idx, open("../data/{}/{}_{}.json".format("ice",
             "ice", "label"), "w"), cls=MyEncoder)
  return id2seqs, topic2id, id2topic

def get_topic_hdict(cand=False):
  if cand :
    cand_topics = ["computers consumer electronics", "arts entertainment", "finance"]
  iceid2topic = "/home/t-dechen/data/ICE/ICEId2Topic.tsv"
  lines = [line.split("/") for line in open(iceid2topic, "r")]
  th = defaultdict(set)
  t_set = set()
  for k,line in enumerate(lines):
    # print("line", line)
    # th["root"].add(line[1].strip())
    line[0] = "root"
    line = [label.strip() for label in line]
    top = line[1]
    if cand and (top not in cand_topics):
      continue
    if cand:
      assert top in cand_topics
    #if k==3: print(line)
    for i in range(len(line)-1):
      par = line[i]
      cld = line[i+1]
      t_set.add(cld)
      th[par].add(cld)
  print("total topics:", len(t_set))
  return th

def get_hf(th, key):
  content = []
  content += [[key.split(",")[-1]]]
  for child in th[key]:
    content += [[key.split(",")[-1] + "/" + item[0]] for item in get_hf(th, child)]
  return content

def gen_hf():
  th = get_hdict()
  content = get_hf(th, "root")
  hf = "../data/ice/label_hierarchical.txt"
  hf = csv.writer(open(hf, "w"))
  hf.writerows(content)

def check():
  iceid2topic = "/home/t-dechen/data/ICE/ICEId2Topic.tsv"
  lines = [line.split("/") for line in open(iceid2topic, "r")]
  kit = []
  cnt = set()
  for line in lines:
    cnt.add(line[0].strip())
    if len(line)==2:
      kit += [line]
      print(line)
  print(len(kit), len(cnt))

def get_candids():
  iceid2topic = "/home/t-dechen/data/ICE/ICEId2Topic.tsv"
  lines = [line.split("/") for line in open(iceid2topic, "r")]
  cand_topics = ["computers consumer electronics", "arts entertainment", "finance"]
  cand_ids = []
  for line in lines:
    line = [_.strip() for _ in line]
    if line[1] in cand_topics:
      if line[0] not in cand_ids:
        cand_ids += [line[0]]
  print(len(cand_ids))
  return cand_ids

def get_layerids():
  if os.path.exists("../data/{}/{}_{}.json".format("ice","ice","layer_ids")):
    layer_ids = json.load(open("../data/{}/{}_{}.json".format("ice","ice","layer_ids"), "r"))["layer_ids"]
    return layer_ids
  id2seqs, topic2id, id2topic = get_idxs()
  layer_ids = defaultdict(list)
  th = get_topic_hdict(cand=True)
  cand_topics = ["computers consumer electronics", "arts entertainment", "finance"]
  d = deque()
  total_size = 0
  showup = set()
  for topic in cand_topics:
    d.append(topic)
    showup.add(topic)
  level = 0
  while d:
    level += 1
    size = len(d)
#    print("size:", size)
    total_size += size
    layers = []
    for i in range(size):
      cur = d.popleft()
      # showup.add(cur)
      layers += [cur]
      for child in th[cur]:
        if child not in showup:
          showup.add(child)
          d.append(child)
    print(len(layers))
    for topic in layers:
      lid = topic2id[topic]
      layer_ids[level] += [lid]
  print(total_size)
  data = {"layer_ids":layer_ids}
  json.dump(data, open("../data/{}/{}_{}.json".format("ice", "ice", "layer_ids"), "w"), cls=MyEncoder)
  return layer_ids

def split_data():
  cand_ids = get_candids()
  data = defaultdict(list)

  filter_doc = open("../data/ice/sample.tsv","r")
  for line in filter_doc:
    label = line.split("\t")[0]
    text = line.split("\t")[1].strip()
    data[label] += [text]
  train, dev, test = [], [], []

  for key, value in data.items():
    print(key, len(value))
    size = len(value)
    train_s, dev_s = math.floor(0.6*size), math.floor(0.2*size)
    test_s = size - train_s - dev_s
    new_line = []
    for val in value:
      new_line += [[key + "\t" + val ]]

    train += new_line[:train_s]
    dev += new_line[train_s:train_s+dev_s]
    test += new_line[train_s+dev_s:]
  print(train[0])
  np.random.shuffle(train)
  np.random.shuffle(test)
  np.random.shuffle(dev)
  train_f = csv.writer(open("../data/ice/train.tsv","w"))
  train_f.writerows(train)
  dev_f = csv.writer(open("../data/ice/dev.tsv","w"))
  dev_f.writerows(dev)
  test_f = csv.writer(open("../data/ice/test.tsv","w"))
  test_f.writerows(test)


def sample_data():
  """
  ['10019\t', 'computers consumer electronics\n']
  ['10013\t', 'arts entertainment\n']
  ['10012\t', 'finance\n']
  """
  cand_ids = get_candids()
  doc = open("/home/t-dechen/data/ICE/ICE-train.tsv", "r")
  filter_doc = open("../data/ice/sample.tsv","w")

  for line in doc:
    label_id = line.split("\t")[0].strip()[9:]
#    print(label_id, label_id in cand_ids)
#    text = line.split("\t")[-1].strip()
    if label_id in cand_ids:
      filter_doc.write(line)

  doc.close()
  filter_doc.close()

def get_word2vec():
  word2idx, idx2word = get_word2idx()
  gen_word2vec(word2idx)

def get_word2idx():
  if os.path.exists("../data/{}/word2idx_{}.json".format("ice", "ice")):
    word2idx = Counter(json.load(open("../data/{}/word2idx_{}.json".format("ice", "ice"), "r"))["word2idx"])
    idx2word = json.load(open("../data/{}/word2idx_{}.json".format("ice", "ice"), "r"))["idx2word"]
    return word2idx, idx2word
  with open("../data/ice/sample.tsv","r") as f:
    text = [line.split("\t")[-1].strip() for line in f]
    docs = [tokenize(_) for _ in text]
  wordDict = Counter()
  word2idx, idx2word = {}, []
  word2idx["u-n-k"] = 0
  word2idx["n-u-l-l"] = 1
  idx2word += ["u-n-k", "n-u-l-l"]
  for doc in docs:
    for token in doc:
      wordDict[token] += 1
  for key,value in wordDict.items():
    if value>2:
      word2idx[key] = len(word2idx)
      idx2word += [key]
  print(len(word2idx))
  assert len(word2idx) == len(idx2word)
  print(idx2word[:10])
  #for i in range(len(idx2word)):
  #  print(idx2word[i], word2idx[idx2word[i]])
  shared = {"word2idx": word2idx, "idx2word":idx2word}
  json.dump(shared, open("../data/{}/word2idx_{}.json".format("ice", "ice"), "w"))
  return word2idx, idx2word

def get_decode_inp(seq, EOS=647):
  decode_inp = []
  pos = len(seq)-1
  for i, idx in enumerate(seq[:-1]):
    if idx==EOS:
      pos = i
    if i < pos : decode_inp += [idx]
    else: decode_inp += [0]    # for pad
  return decode_inp

def get_y_len(seq):
  cnt = 0
  for t_id in seq[1:]:
    if t_id !=0:  cnt += 1
  return cnt

def _process_docs(docs, word2idx, max_docs_length=48):
  docs_lens = [len(doc) for doc in docs]
  # max_docs_length = 0
  for doc in docs:
    max_docs_length = \
        len(doc) if len(doc) > max_docs_length else max_docs_length
  print("max_doclen:", max_docs_length)

  docs2mat = [[word2idx[doc[_]] if _ < len(doc) else 1 for _ in range(max_docs_length)] for doc in docs]
  docs2mask = [[1 if _ < len(doc) else 0 for _ in range(max_docs_length)] for doc in docs]
  docs_lens = [len(doc) if len(doc)<max_docs_length else max_docs_length for doc in docs]
  return docs2mat, docs2mask, docs_lens

def read_data(data_type, word2idx, debug_type=""):
  print("preparing {} data".format(data_type))
  if data_type == "train":
    docs, y_seqs, decode_inps, seq_lens = load_data(data_type, debug_type)
    filter_docs, filter_y_seqs, filter_decode_inps, filter_y_lens\
        =  [], [], [], []
    for doc, y_seq, decode_inp, seq_len in zip(docs,
        y_seqs, decode_inps, seq_lens):
      if len(doc) > 0:
        filter_docs += [doc]
        filter_y_seqs += [y_seq]
        filter_decode_inps += [decode_inp]
        filter_y_lens += [seq_len]
    docs, y_seqs, decode_inps, seq_lens = \
        filter_docs, filter_y_seqs, filter_decode_inps, filter_y_lens
    docs2mat, docs2mask, docs_lens = _process_docs(docs, word2idx)
    data = {
          "raw": docs,
          "x": docs2mat,
          "x_mask":docs2mask,
          "x_len": docs_lens,
          "y_seqs":y_seqs,
          "decode_inps": decode_inps,
          "y_len": seq_lens,
         }
    json.dump(data, open("../data/{}/{}_{}{}.json".format("ice", "ice", data_type, debug_type), "w"), cls=MyEncoder)
    return DataSet(data, data_type)
  else:
    docs, ys = load_data(data_type, debug_type)
    filter_docs, filter_ys = [], []
    for doc, y in zip(docs, ys):
      if len(doc) > 0:
        filter_docs += [doc]
        filter_ys += [y]
    docs, ys = filter_docs, filter_ys
    # to check : some id donot show up
    #mlb = MultiLabelBinarizer()
    #ys = mlb.fit_transform(ys)
    docs2mat, docs2mask, docs_lens = _process_docs(docs, word2idx)
    data = {
         "raw": docs,
         "x": docs2mat,
         "x_mask": docs2mask,
         "x_len": docs_lens,
         "y_h": ys,
         "y_f": ys
        }
    json.dump(data, open("../data/{}/{}_{}{}.json".format("ice", "ice", data_type, debug_type), "w"), cls=MyEncoder)
    return DataSet(data, data_type)

def load_data(data_type, debug_type=""):
  # only need in train
  y_seqs, decode_inps, y_lens = [], [], []
  r_docs, ys = [], []

  id2seqs, topic2id, id2topic = get_idxs()
  EOS = topic2id["end"]
  with open("../data/{}/{}{}.tsv".format("ice", data_type, debug_type),"r") as f:
    lines = [line.split("\t") for line in f]
    labels = [line[0][9:] for line in lines]
    docs = [line[1].strip() for line in lines]
    docs = [doc.split(" ") for doc in docs]

    for label, doc in zip(labels, docs):
      y = []
      seqs = id2seqs[label]  # [start +  label + end  + pad]
      if data_type == "train":
        for seq in seqs:
          r_docs += [doc]
          y_seqs += [seq[1:]]
          decode_inps +=  [get_decode_inp(seq)]  #[seq[:-1]]
          y_lens += [get_y_len(seq)]
      else :
        for seq in seqs:
          for topic_id in seq[1:-1]:
            if topic_id!=0 and topic_id!=EOS: y+= [topic_id]
        ys += [y]
        r_docs += [doc]

  if data_type == "train":
    return r_docs, y_seqs, decode_inps, y_lens
  else:
    return r_docs, ys

def get_data():
  debug_type = ""   #  "_debug"  # debug
  word2idx = Counter(json.load(open("../data/{}/word2idx_{}.json".format("ice", "ice"), "r"))["word2idx"])
  read_data("train", word2idx, debug_type)
  read_data("test", word2idx, debug_type)
  read_data("dev", word2idx, debug_type)

def get_ft(data_type):
  id2seqs, topic2id, id2topic = get_idxs()
  EOS = topic2id["end"]
  with


  lines = [line.split("\t") for line in open("../data/{}/{}.tsv".format("ice", data_type), "r")]
  new_lines = []
  for line in lines:
    nls = set()
    label, text = line[0], line[1]
    for tps in id2seqs[label[9:]]:
      for tp in tps:
        if tp!=1 or tp!=EOS: nls.add(tp)
    newl = ""
    for nl in nls: newl += "__label__" + str(nl) + "\t"
    newl += text
    new_lines += [newl]
  print(new_lines[0])
  with open("../data/{}/{}_{}.tsv".format("ice", data_type, "topic"), "w") as f:
    writer = csv.writer(f)
    writer.writerows(new_lines)

def write_ft():
  get_ft("train")
  get_ft("dev")
  get_ft("test")

if __name__=="__main__":
  #statistic(get_topic_hdict())
  #statistic(get_hdict())
  #gen_hf()
  #check()
  #get_idxs()
  #sample_data()
  # split_data()
  # get_word2idx()
  # get_data()
  # get_word2vec()
  # process_hclf()
  write_ft()
