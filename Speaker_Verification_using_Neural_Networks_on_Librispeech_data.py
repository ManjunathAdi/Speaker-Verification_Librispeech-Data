
##########  Neural Networks for speaker verification is run on
# librispeech's challenging data  and 
#imposter data is obtained from means of UBM(256/512) run on certain speakers of librispeech data
# some times imposter data is selected randomly but this is corrected and made better by using UBM to get imposter data
#Results are at end of the script with details of the parameter-combination run for each of them

print("Bhagavantha Mahamrutyunjaya")

import logging
import os
#from joblib import Memory
import numpy as np
from os import walk
from sklearn.mixture import GaussianMixture as GMM
#from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from python_speech_features import mfcc, logfbank
import scipy.io.wavfile as wav
import time, pickle
import soundfile as sf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.regularizers import l2,l1

    
def ext_features(wavpath):
    "Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
    (sig,rate) = sf.read(wavpath)   
    #clean_signal = remove_silence(rate, sig)
    #mfcc_feat = mfcc(clean_signal,rate)
    mfcc_feat = mfcc(sig,rate,numcep=24)
    return mfcc_feat
   
def train_UBM(M, agg_feats):
    keylist = list(agg_feats.keys())
    ubm_feats = agg_feats[keylist[0]]
    for d in keylist[1:]:
        ubm_feats = np.concatenate((ubm_feats, agg_feats[d]))    
    # === TRAIN UBM ===
    ubm = GMM(n_components=M, covariance_type='full')
    ubm.fit(ubm_feats)

    ubm_params = {}
    ubm_params['means'] = ubm.means_
    ubm_params['weights'] = ubm.weights_
    ubm_params['covariances'] = ubm.covariances_
    ubm_params['precisions'] = ubm.precisions_    
    return ubm_params    
    
script_start_time = time.time()    
       
trainfolder="Downloads/dev-other.tar/train"
ubmfolder="Downloads/dev-other.tar/ubm"
 
 ######################  ubm part              
files,ubmfeatures = [],{}
for (dirpath, dirnames, filenames) in walk(ubmfolder):
    files.extend(filenames)               
ubmfeatures = {wavpath:ext_features(os.path.join(ubmfolder, wavpath)) for wavpath in files}
   

keylist = list(ubmfeatures.keys())
ubm_feats = ubmfeatures[keylist[0]]
for d in keylist[1:]:
    ubm_feats = np.concatenate((ubm_feats, ubmfeatures[d]))

M=512
M=256
D=24

start_time = time.time()
print("\nTraining UBM\n")
ubm_params = train_UBM(M, ubmfeatures)

print("---Time Taken to run UBM = %s seconds ---" % (time.time() - start_time))

filename = '256G_ubmprms_librispeech.sav'
pickle.dump(ubm_params, open(filename, 'wb'))
#fname = '512G_ubmprms_librispeech_.sav'
#params = pickle.load(open(fname, 'rb'))

imposter_data = ubm_params['means']
#imposter_data = params['means']
               
######################### training part
trainingfiles,trainfeatures = [],{}
for (dirpath, dirnames, filenames) in walk(trainfolder):
    trainingfiles.extend(filenames)                 
trainfeatures = {wavpath:ext_features(os.path.join(trainfolder, wavpath)) for wavpath in trainingfiles}
              
                          
tot_num_speech_frames,mun=0,0
aggfeatures = {}
for wavpath, features in trainfeatures.items():
    #label = trainingdata[wavpath]
    normed = features       
    aggfeatures[wavpath.split('-')[0]] = normed    
    #aggfeatures[wavpath] = normed
    tot_num_speech_frames = tot_num_speech_frames + features.shape[0]
    mun=mun+1
 
spklist=list(aggfeatures.keys())

testpath = "Downloads/dev-other.tar/alltestfiles"

f = []
for (dirpath, dirnames, filenames) in walk(testpath):
    f.extend(filenames)
                    
testfeaturesdict={}  
 
for wavpath in f:        
    testfeatures = ext_features(os.path.join(testpath, wavpath)) 
    testfeaturesdict[wavpath] = testfeatures
    
start_time = time.time()

list1,list2,list3=[],[],[]

for spk in spklist:    
    print("\nThe speaker whom we are verifying is: ", spk)
    
    spk1_data=np.zeros((aggfeatures[spk].shape[0], (D+1)))
    spk2_data=np.zeros((imposter_data.shape[0], (D+1)))
    
    spk1_data[:,0:D] = aggfeatures[spk]
    spk1_data[:,D] = 1
    spk2_data[:,0:D] = imposter_data
    spk2_data[:,D] = 0       
    spk_data=np.zeros(((aggfeatures[spk].shape[0]+imposter_data.shape[0]), (D+1)))
    
    spk_data[0:aggfeatures[spk].shape[0],:]=spk1_data
    spk_data[aggfeatures[spk].shape[0]:,:]=spk2_data        
    np.random.shuffle(spk_data)
    
    x_train=spk_data[:,0:D]
    y_train=spk_data[:,D]        
    y_train = y_train.astype('int')
        
    batch_size=50
    epochs=10    
    model = Sequential()
    model.add(Dense(400, activation='relu', input_shape=(D,), W_regularizer = l1(.0001)))
    model.add(Dropout(0.1))
    model.add(Dense(400, activation='relu', W_regularizer = l1(.0001)))
    model.add(Dropout(0.1))
#    model.add(Dense(512, activation='relu', W_regularizer = l1(.0001)))
#    model.add(Dropout(0.2))
#    model.add(Dense(256, activation='relu', W_regularizer = l1(.0001)))
#    model.add(Dropout(0.1))
#    model.add(Dense(128, activation='relu', W_regularizer = l1(.0001)))
#    model.add(Dropout(0.1))
#    model.add(Dense(32, activation='relu', W_regularizer = l1(.0001)))
#    model.add(Dropout(0.05))
    model.add(Dense(1, activation='sigmoid'))
    
    model.summary()    
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0001),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_split=0.1)
        
    ### Testing-    
    
    percent, probab = 0.5, 0.55      
    index,ll=0,[]
    scorelist=[]
    totalpre_count, pre_count, totalrec_count, rec_count = 0, 0, 0, 0
    for wavpath in f:             
        testfeatures = testfeaturesdict[wavpath]                    
        if wavpath.split('-')[0]==spk:
            y_test=np.ones((testfeatures.shape[0],))
            score = model.evaluate(testfeatures, y_test, verbose=0)
            probabs = model.predict(testfeatures, verbose=0)
            
            ylabels=np.zeros((testfeatures.shape[0],1))
            for everyframecount in range(0,testfeatures.shape[0]):
                if probabs[everyframecount]>probab:
                    ylabels[everyframecount]=1
            score = np.count_nonzero(ylabels)
            
            totalrec_count = totalrec_count + 1
            if score>(testfeatures.shape[0]*percent):
                rec_count = rec_count + 1
        else:
            y_test=np.zeros((testfeatures.shape[0],))
            score = model.evaluate(testfeatures, y_test, verbose=0)
            probabs = model.predict(testfeatures, verbose=0)
            
            ylabels=np.zeros((testfeatures.shape[0],1))
            for everyframecount in range(0,testfeatures.shape[0]):
                if probabs[everyframecount]>probab:
                    ylabels[everyframecount]=1
            score = np.count_nonzero(ylabels)            
            
            totalpre_count = totalpre_count + 1
            if score<(testfeatures.shape[0]*percent):
                pre_count = pre_count + 1
        scorelist.append(score)
                   
    print("\nFor speaker:- ", spk, "Precision = ",pre_count/totalpre_count, "and Recall(Accuracy) = ", rec_count/totalrec_count)
        
    list1.append(spk)    
    list2.append(pre_count/totalpre_count)
    list3.append(rec_count/totalrec_count)
        
Dict={'Precision':list2, 'Recall':list3, 'Speaker':list1}

import pandas as pd

df = pd.DataFrame(Dict)

print( df)
print( "Mean Accuracy = ", np.mean(list3))
print( "Mean Precision = ", np.mean(list2))  
print("---Time Taken to run = %s seconds ---" % (time.time() - start_time))



'''  percent=0.5 , probab=0.5
Mean Accuracy =  0.91955169014
Mean Precision =  0.833427570547
'''


'''  percent=0.6 , probab=0.6
Mean Accuracy =  0.763542902412
Mean Precision =  0.992514207287
'''


'''  percent=0.5 , probab=0.6
Mean Accuracy =  0.839880958295
Mean Precision =  0.970410691343
'''



'''  percent=0.6 , probab=0.5

Mean Accuracy =  0.808505753205
Mean Precision =  0.977922246308

'''



'''# percent=0.5, probab=0.55 

Mean Accuracy =  0.891218509633
Mean Precision =  0.965032010825
'''
