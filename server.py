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
            embeddings[word] = list(np.fromstring(f.read(binary_len)))
            # print(word, embeddings[word])
    return embeddings

w2v_file = 'PATH TO GoogleNews-vectors-negative300.bin'
glove_file = 'PATH TO glove.6B.100d.txt'
word2vec_embedding = _load_w2v(w2v_file)
glove_embedding = _load_glove(glove_file, 100)

class Word2VecRequestHandler(pyjsonrpc.HttpRequestHandler):

    @pyjsonrpc.rpcmethod
    def add(self, a, b):
        return [ a + b ] * 300

    @pyjsonrpc.rpcmethod
    def word2vec(self, word):
        range = (-0.01, 0.01)
        shape = (300,)
        if word in word2vec_embedding:
            vec = word2vec_embedding[word]
        else:
            vec = list(np.random.uniform(low=range[0], high=range[1], size=shape))
            word2vec_embedding[word] = vec
        return vec

    @pyjsonrpc.rpcmethod
    def glove(self, word):
        range = (-1, 1)
        shape = (100,)
        if word in glove_embedding:
            vec = glove_embedding[word]
        else:
            vec = list(np.random.uniform(low=range[0], high=range[1], size=shape))
            glove_embedding[word] = vec
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