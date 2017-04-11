import pandas as pd
import numpy as np
import re
import nltk
import json
import string
import time
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

stopwords = set(stopwords.words('english'))


def print_time_status(message, start):
    elapsed = int(round(time.time() - start))
    hours = int(elapsed / 3600)
    minutes = int((elapsed % 3600) / 60)
    seconds = int((elapsed % 3600) % 60)
    print(message + ' {0}H {1}M {2}S'.format(hours, minutes, seconds))


class NewsAnalysis(object):
    def __init__(self, voxfile, jezebelfile):
        print('Reading Data...')
        start = time.time()
        vox, jezebel = self.read(voxfile, jezebelfile)
        print_time_status('Reading Data complete', start)
        print('Cleaning Data...')
        corpus = self.clean(vox, jezebel)
        print_time_status('Cleaning Data Complete', start)
        print('Computing LDA')
        corpus, topic_word_assoc = self.lda(corpus)
        print_time_status('LDA Computation Complete', start)
        print('Computing Latent Semantic Analysis Vectors...')
        corpus = self.lsa3d(corpus)
        print_time_status('LSA Computation Completed', start)
        print('Computing Naive Bayes...')
        corpus = self.naive_bayes(corpus)
        print_time_status('Naive Bays ComputationComplete', start)
        self.write(corpus, topic_word_assoc)

    @staticmethod
    def read(voxfile, jezebelfile):
        vox = pd.read_json('./data/' + voxfile, lines=True)
        jezebel = pd.read_json('./data/' + jezebelfile, lines=True)
        return vox, jezebel

    @staticmethod
    def clean(vox, jezebel):
        jezebel['id'] = jezebel._id.apply(lambda x: x['$oid'])
        jezebel.drop('_id', axis=1, inplace=True)
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
        jezebel.text = jezebel.text.apply(lambda x: filter_func(x))

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

        vox['org'] = 'vox'
        jezebel['org'] = 'jezebel'
        corpus = pd.concat([vox, jezebel], ignore_index=True)
        corpus.columns = vox.columns

        return corpus

    @staticmethod
    def lda(corpus):
        pipeline = Pipeline([
            ('tf', CountVectorizer()),
            ('lda', LatentDirichletAllocation(learning_method='online', max_iter=30, n_topics=27, n_jobs=-1))
        ])

        parameters = {}

        gs = GridSearchCV(pipeline, parameters)
        gs.fit(corpus.text.apply(lambda x: ' '.join(x)))

        topic_word_assoc = []

        def get_top_words(model, feature_names, n_top_words):
            for topic_idx, topic in enumerate(model.components_):
                topic_word_assoc.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

        tf_feature_names = gs.best_estimator_.steps[0][1].get_feature_names()
        lda = gs.best_estimator_.steps[1][1]
        get_top_words(lda, tf_feature_names, 10)
        topic_matrix = np.round(gs.transform(corpus.text.apply(lambda x: ' '.join(x))), 5)
        topic_df = pd.DataFrame(topic_matrix)
        topic_df.columns = ['topic' + str(t) for t in topic_df.columns]
        topic_df['max_category'] = topic_df.apply(lambda x: x.index[x == np.max(x)][0], axis=1)
        topic_columns = topic_df.columns
        corpus_columns = corpus.columns
        corpus = pd.concat([corpus, topic_df], ignore_index=True, axis=1)
        corpus.columns = list(corpus_columns) + list(topic_columns)
        return corpus, topic_word_assoc

    @staticmethod
    def lsa3d(corpus):
        tfidf = TfidfVectorizer()
        tsvd = TruncatedSVD(n_components=3, n_iter=40)
        tfs = tfidf.fit_transform(corpus.text.apply(lambda x: ' '.join(x)))
        lsa = np.round(tsvd.fit_transform(tfs), 3)
        lsadf = pd.DataFrame(lsa)
        corpus_columns = corpus.columns
        lsadf_columns = ['lsa_component0', 'lsa_component1', 'lsa_component2']
        corpus = pd.concat([corpus, lsadf], ignore_index=True, axis=1)
        corpus.columns = list(corpus_columns) + lsadf_columns
        return corpus

    @staticmethod
    def naive_bayes(corpus):
        tfidf = TfidfVectorizer()
        tsvd = TruncatedSVD(n_components=900, n_iter=100)
        tfs = tfidf.fit_transform(corpus.text.apply(lambda x: ' '.join(x)))
        lsa = tsvd.fit_transform(tfs)
        clf = GaussianNB()
        clf.fit(lsa, corpus['org'])
        probs = np.round(pd.DataFrame(clf.predict_proba(lsa)), 3)
        columns = list(corpus.columns) + ['naive_bayes_' + clf.classes_[0], 'naive_bayes_' + clf.classes_[1]]
        corpus = pd.concat([corpus, probs], ignore_index=True, axis=1)
        corpus.columns = columns
        return corpus

    @staticmethod
    def write(corpus, topic_words):
        corpus.drop(['text', 'naive_bayes_jezebel'], axis=1, inplace=True)
        corpus.to_csv('./news_data.csv', index=False)
        with open('./topic_words.json', 'w') as f:
            f.write(json.dumps(topic_words))
            f.close()
        print('Output Written')


if __name__ == '__main__':
    na = NewsAnalysis('vox.jsonl', 'jezebel.jsonl')
