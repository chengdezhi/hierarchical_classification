#!/usr/bin/env python
# coding: utf-8

def get_sdict(s, t):
  #ss = [int(_.strip()) for _ in s.strip().split("\s")]

  print(s)
  sdict = {}
  for x in s:
    sdict[int(x[0])] = int(x[1])
  print(sdict)
  for key in range(3):
    values = t[key+1]
    total = 0
    for val in values:
      total += sdict[val]
    print(key+1, total)

def process():
  stats_1 = [_.strip().split(" ") for _ in open("stats_1","r")]
  stats_2 = [_.strip().split(" ") for _ in open("stats_2","r")]
  tree = {
          1:[2,3,4,5,6,7,8],
          2:[9,10,11,12,13,14,15],
          3:[16,17,18,19]
         }

  train = get_sdict(stats_1, tree)
  test = get_sdict(stats_2, tree)

if __name__=="__main__":
  process()


