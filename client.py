# coding: utf-8
from __future__ import print_function

import requests
import json


class FasttextClient(object):

    def __init__(self):
        self.url = "http://localhost:4000/jsonrpc"
        self.headers = {'content-type': 'application/json'}
        self.id = 0

    def __call__(self, post):
        self.id += 1
        response = requests.post(
            self.url, data=json.dumps(post), headers=self.headers).json()
        # print(response)
        return response['result']

    def get_vector(self, word):
        post = {
            "method": "fasttextserver.get_vector",
            "params": [word],
            "id": self.id,
        }
        response = self(post)
        value = response['value']
        return value

    def similarity(self, wa, wb):
        post = {
            "method": "fasttextserver.similarity",
            "params": [wa, wb],
            "id": self.id,
        }
        response = self(post)
        value = response['value']
        # print(response)
        return value

    def most_similar(self, word):
        post = {
            "method": "fasttextserver.most_similar",
            "params": [word],
            "id": self.id,
        }
        response = self(post)
        value = response['value']
        return value


if __name__ == '__main__':
    fasttext_client = FasttextClient()

    value = fasttext_client.get_vector('de')
    print(value)

    value = fasttext_client.most_similar('de')
    print(value)

    value = fasttext_client.similarity('de', 'a')
    print(value)
