#!/bin/env python3
'''
Script to run model code from https://github.com/caitlinsmith14/gestnet
See the paper https://pages.jh.edu/csmit372/pdf/smithohara_scil2021_paper.pdf
The code was adapted to see if NN's would pay attention to 2 harmony systems
'''
from dev import *
import argparse

def main(args):

    input_file = args.input

    data_stepwise = Dataset(input_file)

    print(f"Training a model with the following articulators: {args.articulators}")

    model = Seq2Seq(training_data=data_stepwise, articulators=args.articulators)

    if args.load == 'none':
        print("Training the model")
        model.train_model(
            training_data=data_stepwise, 
            n_epochs=args.epochs
            )
        model.save()
    else:
        model = Seq2Seq(load=args.load)

    model.evaluate_model(training_data=data_stepwise)
    model.plot_loss

    test = 'test'

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
        '--input', help="(str) the input data file", default='./trainingdata_stepwise_turkish_3_articulators.tsv'
        )
    parser.add_argument('--load', help="(str) the model file to load", default="none", type=str)
    parser.add_argument('--epochs', help="(int) epochs to train the model", default=200, type=int)
    parser.add_argument(
        "--articulators",
        nargs="*", # so that multiple argument can be entered
        type=str,
        default=['la_output', 'tb_output', 'tc_output']
    )
    args = parser.parse_args()
    main(args)
