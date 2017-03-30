import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords

stopwords = set(stopwords.words('english'))
from string import punctuation
import sklearn


class Vox(object):
    def __init__(self):
        vox = self.read()
        vox = self.clean(vox)

    @staticmethod
    def read():
        vox = pd.read_json('./data/vox.jsonl', lines=True)
        return vox

    @staticmethod
    def clean(vox):
        vox['id'] = vox._id.apply(lambda x: x['$oid'])
        vox.drop('_id', axis=1, inplace=True)
        trans = str.maketrans('', '', string.punctuation + '0123456789')
        vox.text = vox.text.apply(
            lambda x: list(filter(lambda w: w not in stopwords, nltk.word_tokenize(x.lower().translate(trans)))))
        vox.text = vox.text.apply(lambda x: list(filter(lambda y: len(y) > 2, x)))
        vox.text = vox.text.apply(lambda x: list(map(lambda y: y.encode('ascii', 'ignore'), x)))
        vox_date = vox.date.apply(
            lambda x: x[:-1] + 'AM' if x.endswith('a') else x[:-1] + 'PM' if x.endswith('p') else x)
        vox_date = vox_date.apply(lambda x: x[:12] if x.endswith('M') else x)

        for row in vox_date.iteritems():
            try:
                if row[1] == 'NULL':
                    vox_date[row[0]] = np.nan
                else:
                    vox_date[row[0]] = pd.to_datetime(row[1])
            except:
                pass

        vox.date = vox_date
        return vox
