
print("Bhagavantha Mahamrutyunjaya")

  
import pandas as pd
import os
#from joblib import Memory
import numpy as np
from os import walk
#from joblib import Parallel, delayed
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import time
import pickle
from keras.layers.normalization import BatchNormalization
import soundfile as sf
from sklearn.mixture import GaussianMixture as GMM
from keras.layers.recurrent import LSTM, GRU
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Bidirectional
from keras.optimizers import RMSprop
from keras.regularizers import l2,l1
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import initializers
from keras.models import load_model    

sequence_length = 8

D=40 # no. of cepstral co-effs
             
def ext_features(wavpath):
    "Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
    global rate,sig
    if ".flac" in wavpath:
        (sig,rate) = sf.read(wavpath) 
    elif ".wav" in wavpath:
        (rate,sig) = wav.read(wavpath)   
    mfcc_feat = mfcc(sig,rate,numcep=D, nfilt=D)
    #mfcc_feat = mfcc(sig,rate,numcep=D)
    mfcc_feat = preprocessing.scale(mfcc_feat)
    return mfcc_feat

def prep_seq_ubm(ubm_feats=None, maxlen=24):
    datadict1,datadict2={},{}
    stk_prices, s_p = [],[]    
    
    stk_prices = ubm_feats    
    s_p = np.array(stk_prices).T    
    nb_samples = stk_prices.shape[0]
    
    X,key= [],[]   
        
    for i in range(maxlen, nb_samples):
        X.append(stk_prices[i-maxlen:i,])
        key.append(stk_prices[i])
    X=np.array(X)
    key=np.array(key)
    
    for j in range(0,key.shape[0]):
        datadict1[j] = X[j,:,:]

    for j in range(0,key.shape[0]):
        datadict2[j] = key[j,:]
   
    return X, datadict1, datadict2    
    
import logging

import imp
imp.reload(logging)
from datetime import datetime

runtime = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
logger = logging.getLogger(__name__)
#fname=r'AD_RNN_LSTM_%s.log'%runtime
logging.basicConfig(filename=r'SpkrVerf_Bi-RNN_LSTM_%s.log'%runtime,
                        filemode='w',
                        level=logging.DEBUG,
                        format='%(levelname)s @ %(asctime)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p') 
logger.debug('The log file Date')           
  
ubmfolder="Documents/linux_scripts/voice/Aug10th_50voices"
    
 ######################  ubm part              
files,ubmfeatures = [],{}
for (dirpath, dirnames, filenames) in walk(ubmfolder):
    files.extend(filenames)               
ubmfeatures = {wavpath:ext_features(os.path.join(ubmfolder, wavpath)) for wavpath in files}
   
keylist = list(ubmfeatures.keys())
ubm_feats = ubmfeatures[keylist[0]]
for d in keylist[1:]:
    ubm_feats = np.concatenate((ubm_feats, ubmfeatures[d]))

ubm_3d_data, datadict1, datadict2 = prep_seq_ubm(ubm_feats, maxlen=sequence_length)   
   
trainfilepath="Documents/linux_scripts/voice/train/yesh"                
######################### training part
 
files,trainfeatures = [],{}
for (dirpath, dirnames, filenames) in walk(trainfilepath):
    files.extend(filenames) 
              
trainfeatures = {wavpath:ext_features(os.path.join(trainfilepath, wavpath)) for wavpath in files}

keylist_ = list(trainfeatures.keys())
train_feats = trainfeatures[keylist_[0]]
for d in keylist_[1:]:
    train_feats = np.concatenate((train_feats, trainfeatures[d]))

#trainfeatures, j, k = prep_seq_ubm(train_feats, maxlen=sequence_length)                 
train_features, j, k = prep_seq_ubm(train_feats, maxlen=sequence_length)

idx = np.random.choice(train_features.shape[0], size=int((train_features.shape[0])/2), replace=False)

trainfeatures = train_features[idx]

idx = np.random.choice(ubm_3d_data.shape[0], size=trainfeatures.shape[0], replace=False)

imposter_data = ubm_3d_data[idx]       
         
aggfeatures = {}            
aggfeatures[keylist_[0].split('-')[0].split('_')[1]] = trainfeatures  

spk=list(aggfeatures.keys())[0]

spk1_data_y=np.ones((aggfeatures[spk].shape[0], 1))
spk2_data_y=np.zeros((imposter_data.shape[0], 1))

X = np.concatenate((aggfeatures[spk], imposter_data ))
Y = np.concatenate((spk1_data_y, spk2_data_y ))
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=42, stratify=Y )

print(x_train.shape, x_val.shape)
print(np.count_nonzero(y_train), np.count_nonzero(y_val))
validationdata = (x_val, y_val)    

y_train = y_train.astype('int')
y_val = y_val.astype('int')
input_shape = (X.shape[1], X.shape[2])
batch_size=32
epochs=500

start_time = time.time()

model = Sequential()

model.add(Bidirectional(LSTM(128, kernel_initializer=initializers.lecun_uniform(254), return_sequences=True, W_regularizer = l1(.001)), input_shape=input_shape))
#model.add(LSTM(256, input_shape=input_shape, kernel_initializer=initializers.lecun_uniform(254), return_sequences=True, W_regularizer = l1(.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Bidirectional(LSTM(256, kernel_initializer=initializers.lecun_uniform(355), return_sequences=True, W_regularizer = l1(.001))))
#model.add(LSTM(128, kernel_initializer=initializers.lecun_uniform(355), return_sequences=True, W_regularizer = l1(.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(256, kernel_initializer=initializers.lecun_uniform(355), return_sequences=True, W_regularizer = l1(.001))))
#model.add(LSTM(128, kernel_initializer=initializers.lecun_uniform(355), return_sequences=True, W_regularizer = l1(.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Bidirectional(LSTM(64, kernel_initializer=initializers.lecun_uniform(355), return_sequences=False)))
#model.add(LSTM(32, kernel_initializer=initializers.lecun_uniform(355), return_sequences=False, W_regularizer = l1(.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(1, kernel_initializer=initializers.lecun_uniform(658)))
model.add(Activation('sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
count=1
file_loc_name = 'Documents/'+'Google_Aug25_bilstmNeurModel'+'_40MFCC_'+str(count)+'_'+spk+'_'+'seqlen'+str(sequence_length)+'_'+'.h5'

chkptr = ModelCheckpoint(filepath=file_loc_name, monitor='val_loss', verbose=1, save_best_only = True)    

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.0,   
                    validation_data=validationdata,  
                    shuffle=True,                                         
                    callbacks=[early_stop, chkptr])


testpath="Documents/linux_scripts/voice/test"

mpath="Documents/Indian_voices_data"

f = []

for (dirpath, dirnames, filenames) in walk(testpath):
    f.extend(filenames)
    

modelname = 'Google_Aug25_bilstmNeurModel_40MFCC_1_yesh_seqlen8_.h5'

logger.debug('\nThe speaker whom we are verifying is: %s' %(spk))

print("************ Speaker =",spk," & the model =",modelname)
logger.debug('************ The model = %s' %(modelname))

model = load_model(os.path.join(mpath,modelname))  
        
    ### Testing-        
testfeaturesdict={}   
for wavpath in f:        
    testfeatures = ext_features(os.path.join(testpath, wavpath)) 
    testfeats, qw, zx = prep_seq_ubm(testfeatures, maxlen=sequence_length)
    testfeaturesdict[wavpath] = testfeats

percent, probab = 0.5, 0.5   
tlist2,tlist3,tlist4,Result,spklabel = [],[],[],[],[]
index,ll=0,[]
scorelist=[]
totalpre_count, pre_count, totalrec_count, rec_count = 0, 0, 0, 0

for wavpath in f:    
    if spk in wavpath:
        spklabel.append(1)
    else:
        spklabel.append(0)
    
    testfeatures = testfeaturesdict[wavpath]
         
    probabs = model.predict(testfeatures, verbose=0)
    
    ylabels=np.zeros((testfeatures.shape[0],1))
    for everyframecount in range(0,testfeatures.shape[0]):
        if probabs[everyframecount]>probab:
            ylabels[everyframecount]=1

    score = np.count_nonzero(ylabels)        
    if score>(testfeatures.shape[0]*percent):
        Authentication=1
        logger.info("Authentication: Success , filename = %s , percent = %f  , score = %d,   testfile-size = %s"%(wavpath,((score/testfeatures.shape[0])*100),score,testfeatures.shape))
        print("Authentication: Success , filename = %s , percent = %f  , score = %d,   testfile-size = %s"%(wavpath,((score/testfeatures.shape[0])*100),score,testfeatures.shape))

    else:
        Authentication=0
        logger.info("Authentication: Fail , filename = %s , percent = %f  , score = %d,   testfile-size = %s"%(wavpath,((score/testfeatures.shape[0])*100),score,testfeatures.shape))
        print("Authentication: Fail , filename = %s , percent = %f  , score = %d,   testfile-size = %s"%(wavpath,((score/testfeatures.shape[0])*100),score,testfeatures.shape))
        
    tlist2.append(score)
    tlist3.append((score/testfeatures.shape[0])*100)
    tlist4.append(testfeatures.shape)
             
    Result.append(Authentication)
     
tDict={'Filename':f, 'Spkrlabel':spklabel, 'Authentication':Result, 'scores':tlist2, 'percent':tlist3, 'testfile_shape':tlist4}

tdf = pd.DataFrame(tDict)

print(np.count_nonzero(tdf.Authentication))
print(tdf[tdf["Authentication"]==1])
print(tdf.sort_values('percent', ascending=False).head(52))

logger.info('No. of authentications = %d'%(np.count_nonzero(tdf.Authentication)))
logger.info('\n\n%s\n'%(tdf[tdf["Authentication"]==1]))
logger.info('\n\n%s\n'%(tdf.sort_values('percent', ascending=False).head(52)))

print("---Time Taken to run = %s seconds ---" % (time.time() - start_time))
logger.debug("---Time Taken to run = %s seconds ---" % (time.time() - start_time))




