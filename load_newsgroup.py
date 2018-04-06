import json, re, numpy
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from collections import Counter
from sklearn.datasets import fetch_20newsgroups as fetch_data
from sklearn.preprocessing import MultiLabelBinarizer
from common import MyEncoder, DataSet

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

def read_news(config, data_type="train", word2idx=None, max_seq_length=3, filter_size=0):
  print("preparing {} data".format(data_type))
  docs, label_seqs, decode_inps, seq_lens, yhs, yfs  = load_hclf(config, data_type=data_type)
  docs = [tokenize(doc) for doc in docs]

  filter_docs, filter_label_seqs, filter_decode_inps, filter_seq_lens, filter_yhs, filter_yfs = [], [], [], [], [], []
  for doc,label_seq,decode_inp,seq_len, yh, yf in zip(docs,label_seqs,decode_inps,seq_lens, yhs, yfs):
    if len(doc)>filter_size:
      filter_docs += [doc]
      filter_label_seqs += [label_seq]
      filter_decode_inps += [decode_inp]
      filter_seq_lens += [seq_len]
      filter_yhs += [yh]
      filter_yfs += [yf]
  docs, label_seqs, decode_inps, seq_lens, yhs, yfs = filter_docs, filter_label_seqs, filter_decode_inps, filter_seq_lens, filter_yhs, filter_yfs

  docs_lens = [len(doc) for doc in docs]
  m1, m2, m3, m4, m5, m6 = 0, 0, 0, 0, 0, 0
  max_docs_length = 0
  average_len = 0.0
  size = len(docs)
  for doc in docs:
    average_len += len(doc)/size
    #print(len(doc))
    if len(doc) > 10000 : m1 += 1
    elif len(doc) > 1000: m2 += 1
    elif len(doc) > 100: m3 += 1
    elif len(doc) > 10: m4 += 1
    elif len(doc) > 0 :
      m5 += 1
      # print(doc)
    else: m6 += 1
    max_docs_length = len(doc) if len(doc) > max_docs_length else max_docs_length
  print(m1, m2, m3, m4, m5, m6)
  print("max_doc_length:", data_type, max_docs_length)
  print("average doc len:", average_len)
  print("max_docs_length:", config.max_docs_length)
  docs2mat = [[word2idx[doc[_]] if _ < len(doc) else 1 for _ in range(config.max_docs_length)] for doc in docs]
  docs2mask = [[1 if _ < len(doc) else 0 for _ in range(config.max_docs_length)] for doc in docs]
  docs_lens = [len(doc) if len(doc)<config.max_docs_length else config.max_docs_length for doc in docs]

  y_seq_mask = [[1 if i<sl else 0 for i in range(max_seq_length)] for sl in seq_lens]
  print(data_type, len(seq_lens))
  data = {
          "raw": docs,
          "x": docs2mat,
          "x_mask":docs2mask,
          "x_len": docs_lens,
          "y_seqs":label_seqs,
          "decode_inps": decode_inps,
          "y_mask": y_seq_mask,
          "y_len": seq_lens,
          "y_h": yhs,
          "y_f": yfs
         }
  # print(data["y_seqs"])
  # for key,val in data.items():
  #   print(key, type(val))

  json.dump(data, open("../data/{}/{}_{}{}.json".format(config.data_from, config.data_from, data_type, config.clftype), "w"), cls=MyEncoder)
  return DataSet(data, data_type)

def load_hclf(config, data_type = "train", max_seqLen = 2):
  seqs = [
          [22,3], [22,4], [22,5], [22,6], [22,7],
          [23,9], [23,10], [23,11], [23,12],
          [24,13], [24,14], [24,15], [24,16],
          [25,8],
          [26,20], [26,18], [26,19],
          [27,21], [27,2], [27,17]
         ]

  tree_map1 = {
             1:[3,4,5,6,7],
             2:[9,10,11,12],
             3:[13,14,15,16],
             4:[8],
             5:[20,18,19],
             6:[21,2,17]
             }
  tree_map2 = {}
  for key,values in tree_map1.items():
    for value in values: tree_map2[value] = key
  news = fetch_data(subset=data_type, remove=('headers', 'footers', 'quotes'))
  docs = news.data
  labels = news.target + 2

  label_seqs = []
  decode_inp = []
  y_h = []
  y_f = []
  seq_len = []
  mlb_1 = MultiLabelBinarizer()
  mlb_1.fit([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]])
  mlb_2 = MultiLabelBinarizer()
  mlb_2.fit([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])

  tree_stas = Counter()

  for label in labels:
    # stastistic
    tree_stas[tree_map2[label]] += 1
    for seq in seqs:
      if label in seq:
        y_h += [seq]
        y_f += [[seq[-1]]]
        if max_seqLen==3:
          label_seqs += [seq+[28]]
          decode_inp += [[1]+seq]
          seq_len += [3]
        else:
          label_seqs += [seq]
          decode_inp += [[1, seq[0]]]
          seq_len += [2]
  print(data_type, len(docs))
  print(tree_stas)
  # if data_type=="test" or config.model_name.endswith("flat"):
  y_h = mlb_1.transform(y_h)
  y_h = [list(_) for _ in list(y_h)]
  y_f = mlb_2.transform(y_f)
  y_f = [list(_) for _ in list(y_f)]
  return docs, label_seqs, decode_inp, seq_len, y_h, y_f

def get_word2idx():
  import cli
  config = cli.config
  train_docs, _, _, _, _, _ = load_hclf(config, "train")
  train_docs = [tokenize(doc) for doc in train_docs]
  test_docs, _, _, _, _, _ = load_hclf(config, "test")
  test_docs = [tokenize(doc) for doc in test_docs]
  docs = train_docs + test_docs
  wordDict = Counter()
  # wordDict["u-n-k"] += 1
  # wordDict["n-u-l-l"] += 1  # for pad
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
  json.dump(shared, open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "w"))

def get_fasttext():
  news = fetch_data(subset="train", remove=('headers', 'footers', 'quotes'))
  docs = news.data
  docs = [" ".join(tokenize(doc)) for doc in docs]
  labels = news.target
  ftrain = open("data/20newsgroup/train.txt", "w")
  for doc, label in zip(docs, labels):
    if len(doc)>0: ftrain.write(doc+"\t__label__"+str(label)+"\n")
  ftrain.close()

  news = fetch_data(subset="test", remove=('headers', 'footers', 'quotes'))
  docs = news.data
  docs = [" ".join(tokenize(doc)) for doc in docs]
  labels = news.target
  ftest = open("data/20newsgroup/test.txt", "w")
  for doc, label in zip(docs, labels):
    if len(doc)>0: ftest.write(doc+"\t__label__"+str(label)+"\n")
  ftest.close()

def get_word2vec():
  import cli
  config = cli.config
  config.max_docs_length = 1000
  word_counter = Counter(json.load(open("data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "r"))["word2idx"])
  word2vec_dict = {}
  if config.pretrain_from == "wiki.en.vec":
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
  json.dump(shared, open("data/{}/word2vec_{}.json".format(config.data_from, config.pretrain_from), "w"))
  print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), w2v_path))


def new_word2idx():
  # has bugs for model training
  import cli
  config = cli.config
  word2vec = Counter(json.load(open("../data/{}/word2vec_{}.json".format(config.data_from, config.pretrain_from), "r"))["word2vec"])
  worddict, word2idx, idx2word = Counter,{}, []
  assert "u-n-k" not in word2vec
  assert "n-u-l-l" not in word2vec
  word2idx["u--n--k"] = 0
  word2idx["n-u-l-l"] = 1
  idx2word += ["u-n-k", "n-u-l-l"]
  for key,value in word2vec.items():
    if key not in word2idx:
      word2idx[key] = len(word2idx)
      idx2word += [key]
  shared = {"word2idx": word2idx, "idx2word": idx2word}
  print("vocab:", len(word2idx))
  json.dump(shared, open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "w"))

def main():
  # log:  change label_len to 2
  import cli, os
  config = cli.config
  config.max_docs_length = 1000
  word2idx = Counter(json.load(open("../data/{}/word2idx_{}.json".format(config.data_from, config.data_from), "r"))["word2idx"])
  train_dict, test_dict = {}, {}
  if os.path.exists("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "train")):
    train_dict = json.load(open("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "train"), "r"))
  if os.path.exists("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "test")):
    test_dict = json.load(open("../data/{}/{}_{}.json".format(config.data_from, config.data_from, "test"), "r"))
  # print(train_dict["x"])
  train_data = DataSet(train_dict, "train") if len(train_dict)>0 else read_news(config, data_type="train", word2idx=word2idx)
  for key, val in train_data.data.items():
    print(key, len(val), val[0])
    #if isinstance(val[0], list) and len(val[0]) > 10: print(val[0][:100])
    #else: print(val[0])
  test_data = DataSet(test_dict, "test") if len(test_dict)>0 else read_news(config, data_type="test", word2idx=word2idx)
  for key, val in test_data.data.items():
    print(key, len(val), val[0])
    #if isinstance(val[0], list) and len(val[0]) > 10: print(val[0][:100])
    #else: print(val[0])

if __name__=="__main__":
  # get_fasttext()
  get_word2idx()
  # main()
  # get_word2vec()
  #new_word2idx()
