import argparse
import utils


def create_parser():
    parser = argparse.ArgumentParser(description='Options for analysis')
    parser.add_argument('-d', '--dataset', default='dataset/')
    return parser




if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    print(args.dataset)
    import os
    for i in os.listdir(args.dataset):
        print(i)
    eeg_full_data = utils.load_eeg()
    print('Data loaded')
