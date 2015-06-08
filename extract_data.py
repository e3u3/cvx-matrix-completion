import urllib2
import numpy as np
import json
import gzip

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
user_ids = []
item_ids = []
ratings = []
count = 0

for review in parse('/home/linuxthink/data/aggressive_dedup.json.gz'):
    count += 1
    if count % 1000000 == 0:
        print count
    user_ids.append(review['reviewerID'])
    item_ids.append(review['asin'])
    ratings.append(review['overall'])

import cPickle as pickle
pickle.dump(user_ids, open("user_ids.data", "wb"))
pickle.dump(item_ids, open("item_ids.data", "wb"))
pickle.dump(ratings, open("ratings.data", "wb"))
