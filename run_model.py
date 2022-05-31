#!/bin/env python3
'''
Script to run model code from https://github.com/caitlinsmith14/gestnet
See the paper https://pages.jh.edu/csmit372/pdf/smithohara_scil2021_paper.pdf
The code was adapted to see if NN's would pay attention to 2 harmony systems
'''
from dev import *
import argparse
import pandas as pd
import statsmodels.api as sm

def analyze(data):
    '''Analyze the data'''
    return

def main(args):

    input_file = args.input

    data_stepwise = Dataset(input_file)

    model = Seq2Seq(training_data=data_stepwise)

    if args.load == 'none':
        model.train_model(training_data=data_stepwise, n_epochs=200)
        model.save()
    else:
        model = Seq2Seq(load=args.load)

    test = 'test'

    print("Evaluating model... Enter n when done to continue")

    while test != 'n':
        model.plot_loss()
        test = input("View model output again? (y/n): ")

    print("Test the model on different data. Enter 'done' when you want to exit")

    while test != 'done':
        test = input("Test word: ")
        if test in data_stepwise.vocabulary:
            model.evaluate_word(training_data=data_stepwise, word=test)
        elif test == 'done':
            break
        else:
            print("Input not accepted... please try again")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', help="(str) the input data file", default='./trainingdata_stepwise_turkish_extended.tsv'
        )
    parser.add_argument('--load', help="(str) the model file to load", default="none")
    args = parser.parse_args()
    main(args)
