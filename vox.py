import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stopwords = set(stopwords.words('english'))

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
        stemmer = SnowballStemmer('english')

        def filter_func(document):
            result = []
            wordlist = nltk.word_tokenize(document.lower().translate(trans))
            for word in wordlist:
                c1 = word not in stopwords
                c2 = len(word) > 2
                c3 = not word.startswith('https')
                c4 = not re.match('document[a-z]+', word)
                if c1 and c2 and c3 and c4:
                    result.append(stemmer.stem(word.encode('ascii', 'ignore').decode('UTF-8')))
            return result

        vox.text = vox.text.apply(lambda x: filter_func(x))
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
