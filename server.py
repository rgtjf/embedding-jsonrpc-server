# coding: utf-8
from __future__ import print_function
import argparse
import pyjsonrpc
import numpy as np


def _load_glove(in_file, dim=100):
    embeddings = {}
    for line in open(in_file).readlines():
        sp = line.split()
        assert len(sp) == dim + 1
        embeddings[sp[0]] = [float(x) for x in sp[1:]]
    return embeddings


def _load_w2v(in_file):
    """
    	Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    embeddings = {}
    with open(in_file, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            embeddings[word] = np.fromstring(f.read(binary_len), dtype='float32').tolist()
            # print(word, embeddings[word])
    return embeddings

''' input file path '''
w2v_file = '/home/junfeng/word2vec/GoogleNews-vectors-negative300.bin'
glove_file = '/home/junfeng/GloVe/glove.6B.100d.txt'

''' word2vec embedding dict and unk word embedding dict '''
word2vec_embedding = _load_w2v(w2v_file)
word2vec_unk_embedding = {}

''' glove embedding dict and unk word embedding dict '''
glove_embedding = _load_glove(glove_file, 100)
glove_unk_embedding = {}

class Word2VecRequestHandler(pyjsonrpc.HttpRequestHandler):

    @pyjsonrpc.rpcmethod
    def add(self, a, b):
        return [ a + b ] * 300

    @pyjsonrpc.rpcmethod
    def word2vec(self, word):
        range = (-0.25, 0.25)
        shape = (300,)
        if word in word2vec_embedding:
            vec = ('in', word2vec_embedding[word])
        elif word in word2vec_unk_embedding:
            vec = ('unk', word2vec_unk_embedding[word])
        else:
            vec = list(np.random.uniform(low=range[0], high=range[1], size=shape))
            word2vec_unk_embedding[word] = vec
            vec = ('unk', vec)
        print('query word2vec', word, vec[0])
        return vec

    @pyjsonrpc.rpcmethod
    def glove(self, word):
        range = (-1, 1)
        shape = (100,)
        if word in glove_embedding:
            vec = ('in', glove_embedding[word])
        elif word in glove_unk_embedding:
            vec = ('unk', glove_unk_embedding[word])
        else:
            vec = list(np.random.uniform(low=range[0], high=range[1], size=shape))
            glove_unk_embedding[word] = vec
            vec = ('unk', glove_unk_embedding[word])
        print('query glove', word, vec[0])
        return vec

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A embedding JSON/RPC server')
    # parser.add_argument('model', metavar='MODEL',
    #                     help='word2vec model in binary formmat')
    parser.add_argument('port',
                        type=int,
                        default=8083,
                        help='port number')
    args = parser.parse_args()


    http_server = pyjsonrpc.ThreadingHttpServer(
        server_address=('localhost', args.port),
        RequestHandlerClass=Word2VecRequestHandler)

    print("Starting word2vec server ...")
    print("URL: http://localhost:{}".format(args.port))
    http_server.serve_forever()