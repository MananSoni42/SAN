#!/usr/bin/env bash

# Train without classifier with lambda = 1.5 for 10 epochs
python3 train.py --v2_on --classifier_gamma=1.5 --epochs=10

# Train with classifier with lambda = 0.1 for 50 epochs
python3 train.py --v2_on --classifier_on --classifier_gamma=0.1 --epochs=50

