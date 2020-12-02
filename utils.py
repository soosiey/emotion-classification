import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA,FastICA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

WHEEL_MAP = {'Pride':1,'Elation':2,'Joy':3,'Satisfaction':4,
             'Relief':5,'Hope':6,'Interest':7,'Surprise':8,
             'Sadness':9,'Fear':10,'Shame':11,'Guilt':12,
             'Envy':13,'Disgust':14,'Contempt':15,'Anger':16}
def load_eeg(d):

    data = []
    labels = []
    for f in tqdm(range(1,33)):
        if(f < 10):
            s = '0'+str(f)
        else:
            s = str(f)
        tmp = pickle.load(open(d+'data_preprocessed_python/'+'s'+s+'.dat','rb'),encoding='latin1')
        data.append(tmp['data'])
        labels.append(tmp['labels'])
    ret = {'data':np.concatenate(data,axis=0),'labels':np.concatenate(labels,axis=0)}


    return ret

def load_meta(d):
    return pd.read_csv(d + 'metadata/participant_ratings.csv'), pd.read_csv(d + 'metadata/online_ratings.csv')

def preprocess_eeg(d):
    d_shape = d.shape
    ret = np.zeros((d_shape[0],d_shape[1] * d_shape[2]))
    for i in range(d_shape[0]):
        temp = np.zeros(d_shape[1] * d_shape[2])
        for j in range(d_shape[1]):
            temp[j * d_shape[2]:(j+1) * d_shape[2]] = d[i,j]
        ret[i] = temp
    return ret

def dim_reduce(data,type='PCA', n_comps=32):
    ret = None
    if(type == 'PCA'):
        pca = PCA(n_components=n_comps)
        pca.fit(data)
        tot = np.sum(pca.singular_values_)
        print('First 13 component:',np.sum(pca.singular_values_[0:13])/tot)
        tmp = pca.components_
        ret = tmp,pca.transform(data)
    if(type == 'ICA'):
        ica = FastICA(n_components=n_comps)
        ica.fit(data)
        tmp = ica.components_
        ret = tmp,ica.transform(data)
    if(type == 'TSNE'):
        pca = PCA(n_components=n_comps)
        tsne = TSNE(n_components=2,early_exaggeration=20,learning_rate=1000,init='random',verbose=2,n_jobs=8,perplexity=50)
        x = pca.fit_transform(data)
        tsne.fit(x)
        tmp = tsne.embedding_
        ret = tmp,tsne.fit_transform(x)
    return ret

def view_components(data):
    n_comps,n_feats = data.shape[0],data.shape[1]
    components = np.zeros((n_comps,40,n_feats//40))
    for i in range(n_comps):
        components[i] = data[i].reshape((40,n_feats//40))
    plt.figure(figsize=(20,20))
    for i in range(n_comps):
        plt.subplot(10,10,i+1)
        tmp = components[i]
        tmp = tmp / np.amax(tmp)
        tmp = tmp.astype(int)
        plt.pcolormesh(tmp)
    plt.show()

def view_features(data):
    n_samp,n_feats = data.shape[0],data.shape[1]
    features = np.zeros((min(100,n_samp),8,n_feats//8))
    for i in range(min(100,n_samp)):
        features[i] = data[i].reshape((8,n_feats//8))
    print(features.shape)
    plt.figure(figsize=(20,20))
    for i in range(min(100,n_samp)):
        plt.subplot(10,10,i+1)
        tmp = features[i]
        tmp = tmp / np.amax(tmp)
        tmp = tmp.astype(int)
        plt.pcolormesh(tmp)
    plt.show()

def view_features_tsne(data,labels):
    plt.figure(figsize=(20,20))
    tmp = data[labels[:,0] < 4]
    plt.scatter(*tmp.T,c='red')
    tmp = data[((labels[:,0] >= 4) & (labels[:,0] < 7))]
    plt.scatter(*tmp.T,c='green')
    tmp = data[labels[:,0] >= 7]
    plt.scatter(*tmp.T,c='blue')
    plt.show()

def split_data():
    data = load_eeg('dataset/')
    d,l =  data['data'],data['labels'] # preprocess_eeg(data['data']), data['labels']

    tmp = np.zeros((d.shape[0],d.shape[2],d.shape[1]))
    for i in range(tmp.shape[0]):
        tmp[i] = d[i].T
    d = tmp
    xtrain,xtest,ytrain,ytest = train_test_split(d,l,test_size=.25)
    ytrain = np.around(ytrain).astype(int)
    ytest = np.around(ytest).astype(int)

    np.save('xtrain.npy', xtrain)
    np.save('xtest.npy', xtest)
    np.save('ytrain.npy', ytrain)
    np.save('ytest.npy', ytest)
if __name__ == '__main__':
    print('Testing eeg loading...')
    data = load_eeg('dataset/')
    print('Data length:', len(data))
    print('Data keys:', data.keys())
    for key in data.keys():
        print(key,'shape:',data[key].shape)

    print('Testing metadata loading...')
    meta_p, meta_o = load_meta('dataset/')
    print(meta_p.head())
    print(meta_p.shape)
    print(meta_o.head())
    print(meta_o.shape)
    print(meta_p)

    print('Testing dim reductions...')
    d,l = preprocess_eeg(data['data']),data['labels']
    print(d.shape,l.shape)
    c,new_data = dim_reduce(d,'TSNE')
    print(c.shape,new_data.shape,l.shape)
    #view_components(c)
    #view_features(new_data)
    view_features_tsne(new_data,l)
