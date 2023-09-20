#!/usr/bin/env python3

import sys

from test_pooling import *
from test_conv2d import *
import argparse

sys.path.append('autograder')
from helpers import *


parser = argparse.ArgumentParser(description='Run the hw2 bonus autograder')
parser.add_argument('-s', '--summarize', action='store_true',
                    help='only show summary of scores')
args = parser.parse_args()

################################################################################
#################################### DO NOT EDIT ###############################
################################################################################

tests = [
    {
        'name': 'Question 1.1 - Conv2D Forward',
        'autolab': 'Conv2d Forward',
        'handler': test_conv2d_forward,
        'value': 2,
    },
    {
        'name': 'Question 1.2 - Conv2D Backward',
        'autolab': 'Conv2d Backward',
        'handler': test_conv2d_backward,
        'value': 3,
    },
    {
        'name': 'Question 2.1 - MaxPool2D Forward',
        'autolab': 'MaxPool2d Forward',
        'handler': test_maxpool2d_forward,
        'value': 2,
    },
    {
        'name': 'Question 2.2 - MaxPool2D Backward',
        'autolab': 'MaxPool2d Backward',
        'handler': test_maxpool2d_backward,
        'value': 3,
    },
    {
        'name': 'Question 3.1 - AvgPool2D Forward',
        'autolab': 'AvgPool2d Forward',
        'handler': test_avgpool2d_forward,
        'value': 2,
    },
    {
        'name': 'Question 3.2 - AvgPool2D Backward',
        'autolab': 'AvgPool2d Backward',
        'handler': test_avgpool2d_backward,
        'value': 3,
    },
]

tests.reverse()

if __name__ == '__main__':
    np.random.seed(2020)
    run_tests(tests, summarize=args.summarize)
