# embedding-jsonrpc-server
help to reduce the time to load word2vec or other embeddings

## Ref
- https://pypi.python.org/pypi/python-jsonrpc

## Pre-Requirement
- jsonrpc

  ```bash
  $ pip install python-jsonrpc
  ```
- Word2Vec
- GloVe
  
  ```
  # server.py line 39, 40
  w2v_file = 'PATH TO GoogleNews-vectors-negative300.bin'
  glove_file = 'PATH TO glove.6B.100d.txt'
  ```

## Server

```
$ python server.py
```

## Client

```
$ python clint.py
```