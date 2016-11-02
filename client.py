# coding: utf-8
from  __future__ import print_function
import pyjsonrpc

class Embedding(object):

    def __init__(self):
        self.http_client = pyjsonrpc.HttpClient(
            url = "http://localhost:8083",
        )

    def get_word2vec(self, word):
        vec = self.http_client.word2vec(word)
        return vec

    def get_glove(self, word):
        vec = self.http_client.glove(word)
        return vec

embedding = Embedding()

if __name__ == '__main__':

    http_client = pyjsonrpc.HttpClient(
        url="http://localhost:8083",
    )


    c = http_client.add(1, 2)
    print(c, type(c))

    word2vec = http_client.word2vec("cat")
    print(word2vec, len(word2vec[1]))

    glove = http_client.glove("cat")
    print(glove)
