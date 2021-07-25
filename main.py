import argparse
import utils
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser(description='Options for analysis')
    parser.add_argument('-d','--dataset',default='dataset/')
    parser.add_argument('-t','--type',default='PCA')
    parser.add_argument('-n','--num',default=32)
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    eeg_full_data = utils.load_eeg(args.dataset)
    meta_p, meta_o = utils.load_meta(args.dataset)

    print('Data loaded')
    data = utils.preprocess_eeg(eeg_full_data['data'])
    labels = np.around(eeg_full_data['labels']).astype(int)

    print('Performing Reductions...')
    components, new_data = utils.dim_reduce(data,args.type,n_comps=int(args.num))

    print('Showing features...')
    utils.view_features(new_data)