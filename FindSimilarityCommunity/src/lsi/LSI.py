# import modules
import gensim
import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

"""
LSI 
"""


class LSI:
    @staticmethod
    def load_data(path, file_name):
        """
        Input  : path and file_name
        Purpose: loading text file
        Output : list of paragraphs/documents and
                 title(initial 100 words considred as title of document)
        """
        documents_list = []
        titles = []
        with open(os.path.join(path, file_name), "r") as fin:
            for line in fin.readlines():
                text = line.strip()
                documents_list.append(text)
        print("Total Number of Documents:", len(documents_list))
        titles.append(text[0:min(len(text), 100)])
        return documents_list, titles

    @staticmethod
    def preprocess_data(doc_set):
        """
        Input  : docuemnt list
        Purpose: preprocess text (tokenize, removing stopwords, and stemming)
        Output : preprocessed text
        """
        # initialize regex tokenizer

        nltk.download('stopwords')
        tokenizer = RegexpTokenizer(r'\w+')
        # create English stop words list
        en_stop = set(stopwords.words('english'))
        # Create p_stemmer of class PorterStemmer
        p_stemmer = PorterStemmer()
        # list for tokenized documents in loop
        texts = []
        # loop through document list
        for i in doc_set:
            # clean and tokenize document string
            raw = i.lower()
            tokens = tokenizer.tokenize(raw)
            # remove stop words from tokens
            stopped_tokens = [i for i in tokens if not i in en_stop]
            # stem tokens
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            # add tokens to list
            texts.append(stemmed_tokens)
        return texts

    @staticmethod
    def prepare_corpus(doc_clean):
        """
        Input  : clean document
        Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
        Output : term dictionary and Document Term Matrix
        """
        # Creating the term dictionary of our courpus,
        #  where every unique term is assigned an index.
        # dictionary = corpora.Dictionary(doc_clean)
        dictionary = corpora.Dictionary(doc_clean)
        # Converting list of documents
        #  (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
        # generate LDA model
        return dictionary, doc_term_matrix

    @staticmethod
    def create_gensim_lsa_model(doc_clean, number_of_topics):
        """
        Input  : clean document, number of topics and number of words associated with each topic
        Purpose: create LSA model using gensim
        Output : return LSA model
        """
        dictionary, doc_term_matrix = LSI.prepare_corpus(doc_clean)
        # generate LSA model
        lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
        V = gensim.matutils.corpus2dense(lsamodel[doc_term_matrix],
                                         len(lsamodel.projection.s)).T / lsamodel.projection.s
        return V

    @staticmethod
    def getT(path, fileName, numtopics):
        documents_list, titles = LSI.load_data(path, fileName)
        doc_clean = LSI.preprocess_data(documents_list)
        t = LSI.create_gensim_lsa_model(doc_clean, numtopics)
        return t


if __name__ == '__main__':
    LSI.getT(r'D:\workspace\pycharm\paper_algorithm\FindSimilarityCommunity\src\lsi', 'test.txt',10)
