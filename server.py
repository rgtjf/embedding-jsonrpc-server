# coding: utf-8
from __future__ import print_function

import logging

from werkzeug.wrappers import Request, Response
from werkzeug.serving import run_simple
from jsonrpc import JSONRPCResponseManager, dispatcher

from gensim.models.keyedvectors import KeyedVectors


logger = logging.getLogger(__name__)


@dispatcher.add_class
class FasttextServer(object):

    def __init__(self):

        ''' input file path '''
        # english_file = '/home/tjf141457/CIKM AnalytiCup 2018/data/raw/wiki.en.vec'
        spanish_file = '/home/tjf141457/CIKM AnalytiCup 2018/data/raw/wiki.es.1000.vec'

        ''' word2vec embedding dict and unk word embedding dict '''
        # Ref: https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.load_fasttext_format
        self.word_vectors = KeyedVectors.load_word2vec_format(spanish_file, binary=False)
        self.vocab = self.word_vectors.vocab

    def get_vector(self, word):
        response = {
            'msg': 'get_vector',  # 'unk'
            'value': None,
            'params': word
        }
        if word not in self.vocab:
            response['msg'] = 'unk'
        else:
            response['value'] = self.word_vectors[word]
        response['value'] = response['value'].tolist()
        logger.info('{}, {}'.format(response['msg'], response['params']))
        return response

    def most_similar(self, word):
        response = {
            'msg': 'most_similar',
            'value': None,
            'params': word
        }
        if word not in self.vocab:
            response['msg'] = 'unk'
        else:
            response['value'] = self.word_vectors.most_similar(word)
        response['value'] = response['value']
        logger.info('{}, {}'.format(response['msg'], response['params']))
        return response

    def similarity(self, wa, wb):
        response = {
            'msg': 'similarity',
            'value': -1,
            'params': (wa, wb)
        }
        if wa not in self.vocab:
            response['msg'] = 'unk: {}'.format(wa)
        elif wb not in self.vocab:
            response['msg'] = 'unk: {}'.format(wb)
        else:
            response['value'] = float(self.word_vectors.similarity(wa, wb))
        print(response['value'])
        logger.info('{}, {}, {}'.format(response['msg'], response['params'], response['value']))
        return response


@Request.application
def application(request):
    response = JSONRPCResponseManager.handle(
        request.get_data(cache=False, as_text=True), dispatcher)
    return Response(response.json, mimetype='application/json')


if __name__ == '__main__':
    run_simple('localhost', 4000, application)
