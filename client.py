# coding: utf-8
from  __future__ import print_function
import pyjsonrpc
import json

http_client = pyjsonrpc.HttpClient(
    url = "http://localhost:8083",
)


c = http_client.add(1, 2)
print(c, type(c))

word2vec = http_client.word2vec("cat")
print(word2vec)

glove = http_client.glove("cat")
print(glove)