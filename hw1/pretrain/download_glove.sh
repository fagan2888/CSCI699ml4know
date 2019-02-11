#!/usr/bin/env bash
wget http://nlp.stanford.edu/data/glove.6B.zip
mkdir glove.6B
unzip glove.6B.zip -d glove.6B/
rm -rf glove.6B.zip