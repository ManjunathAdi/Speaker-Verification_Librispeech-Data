
##########  SVM-GMM-UBM for speaker verification on librispeech's speech data

print("Bhagavantha Mahamrutyunjaya")

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
import time, pickle, random, shutil
import soundfile as sf

def check_threshold(y_probability, y_actual, threshold, positive_class,
    negative_class):
    y_pred = [positive_class if y >= threshold else negative_class
     for y in y_probability]
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    #fscore = f1_score(y_actual, y_pred)
    #recall = precision_score(y_actual, y_pred)
    #precision = recall_score(y_actual, y_pred)
    return threshold, accuracy, precision, recall

def sweep_thresholds(y_probability, y_actual, start, stop, step,
    positive_class=1, negative_class=0):
    sweep = [ check_threshold(y_probability, y_actual, t,
     positive_class, negative_class) for t in np.arange(start, stop, step)]
    return np.array(sweep)

def get_best_threshold(sweep):
    max_recall = np.max(sweep[:, 2])
    best_recall = sweep[sweep[:, 2] == max_recall]
    #print(best_recall)
    max_precision = np.max(best_recall[:,3])
    best_threshold = best_recall[best_recall[:,3] == max_precision]
    return best_threshold
    
def get_best_fscore(sweep):
    max_fscore = np.max(sweep[:, 4])
    best_fscore = sweep[sweep[:, 4] == max_fscore]
    return best_fscore

    
def ext_features(wavpath):
    "Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
    (sig,rate) = sf.read(wavpath)   
    #clean_signal = remove_silence(rate, sig)
    #mfcc_feat = mfcc(clean_signal,rate)
    mfcc_feat = mfcc(sig,rate,numcep=24)
    return mfcc_feat

def train_UBM(M, agg_feats, num_speech_frames):
    keylist = list(agg_feats.keys())
    ubm_feats = agg_feats[keylist[0]]
    for d in keylist[1:]:
        ubm_feats = np.concatenate((ubm_feats, agg_feats[d]))

    print( "total feats: ", num_speech_frames)   
    # === TRAIN UBM ===
    ubm = GMM(n_components=M, covariance_type='full')
    ubm.fit(ubm_feats)

    ubm_params = {}
    ubm_params['means'] = ubm.means_
    ubm_params['weights'] = ubm.weights_
    ubm_params['covariances'] = ubm.covariances_
    ubm_params['precisions'] = ubm.precisions_    
    return ubm_params    
   
def adapt_UBM_to_speakers(M, speaker_feats, ubm_params):
    # === ADAPT UBM, CONSTRUCT TRAINING FEATURES ===
    speaker_svm_feats = {}      
    for key, feats in aggfeatures.items():
        #print key, feats.shape   
        updated_means = adapt_UBM(M, ubm_params, feats)
        speaker_svm_feats[key] = updated_means

    return speaker_svm_feats
    
def adapt_UBM(M, ubm_params, data):
    updated_means = np.array(ubm_params['means'], dtype=np.float32)

    for it in range(1): # adaptation loop
        gmm = GMM(n_components=M, weights_init=ubm_params['weights'], means_init=updated_means, \
                  precisions_init=ubm_params['precisions'], covariance_type='full')
        gmm.fit(data)    
        #  ==== Actual adaptation code =====
        new_means = gmm.means_
        new_weights = gmm.weights_
        T = data.shape[0]
        updated_means = adapt_means(ubm_params['means'], ubm_params['covariances'],\
                                    ubm_params['weights'], new_means, new_weights, T).flatten('C')

    return updated_means
    
def adapt_means(ubm_means, ubm_covars, ubm_weights, new_means, new_weights, T):
    n_i = new_weights*T
    alpha_i = n_i/(n_i+10)
    new_means[np.isnan(new_means)] = 0.0
    return_means = (alpha_i*new_means.T+(1-alpha_i)*ubm_means.T).T
    
    diag_covars = np.diagonal(ubm_covars, axis1=1, axis2=2)
    
    return_means = ( np.sqrt(ubm_weights) * (1/np.sqrt( diag_covars.T ) ) * return_means.T ).T
    #print return_means.shape
    return return_means
    
script_start_time = time.time()    
        
wavfolder="Downloads/librispeech_data/train-other-500"
trainfolder="Downloads/librispeech_data/train"
ubmfolder="Downloads/librispeech_data/train"

 ######################  ubm part              
files,ubmfeatures = [],{}
for (dirpath, dirnames, filenames) in walk(ubmfolder):
    files.extend(filenames)               
ubmfeatures = {wavpath:ext_features(os.path.join(ubmfolder, wavpath)) for wavpath in files}
   
######################### training part
trainingfiles,trainfeatures = [],{}
for (dirpath, dirnames, filenames) in walk(trainfolder):
    trainingfiles.extend(filenames)                 
trainfeatures = {wavpath:ext_features(os.path.join(trainfolder, wavpath)) for wavpath in trainingfiles}
              
tot_num_speech_frames,mun=0,0
aggfeatures, agg_features = {}, {}
for wavpath, features in trainfeatures.items():
    #label = trainingdata[wavpath]
    normed = features
    agg_features[wavpath.split('-')[0]] = normed
    aggfeatures[wavpath] = normed
    tot_num_speech_frames = tot_num_speech_frames + features.shape[0]
    mun=mun+1
   
spk_list=list(agg_features.keys())

###########################  Train UBM
M=24
D=24

start_time = time.time()
print("\nTraining UBM\n")
ubm_params = train_UBM(M, ubmfeatures, tot_num_speech_frames)

print("---Time Taken to run UBM = %s seconds ---" % (time.time() - start_time))

dumppath="Downloads/librispeech_data"

filename = '24G_ubmprms_24ceps_librispeechdata_.sav'
pickle.dump(ubm_params, open(os.path.join(dumppath,filename), 'wb'))

# load the model from disk
#params = pickle.load(open(filename, 'rb'))

start_time = time.time()
svm_train_features = adapt_UBM_to_speakers(M, aggfeatures, ubm_params) 
print("---Time Taken to run adapt_UBM_to_speakers = %s seconds ---" % (time.time() - start_time))
        
testpath = "Downloads/librispeech_data/test"

f = []
for (dirpath, dirnames, filenames) in walk(testpath):
    f.extend(filenames)        
    
print("\nTest data feature extraction and MAP adaptation \n")    
completemeans = np.zeros((len(f), M*D)) 
index,ll=0,[]
for wavpath in f:        
        testfeatures = ext_features(os.path.join(testpath, wavpath))
        if testfeatures.shape[0]<128:
            print( wavpath,testfeatures.shape)
            ll.append(wavpath)        
        spkmeans = adapt_UBM(M, ubm_params, testfeatures)
        completemeans[index] = spkmeans.reshape(1, M*D)
        index=index+1
        
print("\n Train data shape = ",len(svm_train_features),"*", M*D)      
print("\nTest data shape", completemeans.shape)        
print("\n Testing the test speaker data \n")        
        
print_result=[]       

start_time = time.time()    
fullresultlist=[]    
    

NoOfTestSamples=[]
spk_count=1
for speaker in spk_list:  
    print(spk_count)
    spk_count = spk_count + 1
##    print( "Speaker of interest we are verifying is: ",speaker)
    svm_train_feats = np.zeros((len(svm_train_features), M*D))
    svm_train_labels = np.ones(len(svm_train_features), dtype=np.float32)    
    y_actual = np.zeros(len(f), dtype=np.float32)
    y_pred = np.zeros(len(f), dtype=np.float32)
    y_confidence = np.zeros(len(f), dtype=np.float32)
    #################################### Automating for all speakers:-    
    i,lablist = 0,[]
    for key, features_ in svm_train_features.items():
        svm_train_feats[i] = features_.reshape(1, M*D)
        lablist.append(key)
        if key.split('-')[0] == speaker:        # 
            svm_train_labels[i]=1
            #print '1', key
        else:
            svm_train_labels[i]=0
            #print '-1',key
        i=i+1        
    
    ############################# --- Train SVM ---
    svmclf = svm.SVC(C=1,kernel='linear', probability=True)
#    svmclf = svm.SVC(C=10,kernel='linear', probability=True, class_weight={1: 100})

    #svmclf = CalibratedClassifierCV(svmclf_, method='isotonic')
    #svmclf = CalibratedClassifierCV(svmclf_, method='sigmoid')
    svmclf.fit(svm_train_feats, svm_train_labels)   
    
    ###################   Testing   #####################        
    resultlist,ind=[],0    
    ##################    TESTING All test files in a loop    
    for wavpath in f:
        if wavpath.split('-')[0] == speaker:        
            y_actual[ind]=1            
        else:
            y_actual[ind]=0
         
        ind=ind+1        
    y_pred = svmclf.predict(completemeans)
    proba = svmclf.predict_proba(completemeans)
    NoOfTestSamples.append(int(np.sum(y_actual)))
        
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred )
        
    fullresultlist.append(y_pred)    
    y_prob = proba[:,1]
    out = sweep_thresholds(y_prob, y_actual, 0.1, 0.99, 0.01)    
    best_thresholds = get_best_threshold(out)
    print_result.append(list(best_thresholds[0,]))
    
print("---Time Taken to run = %s seconds ---" % (time.time() - start_time))

threshold, pre, rec, fsco = [],[],[],[]

for d in range(0,len(print_result)):
    pre.append(print_result[d][2])
    rec.append(print_result[d][3])
    #fsco.append(print_result[d][4])
    threshold.append(print_result[d][0])

Dict={'Speaker':spk_list, 'Precision':pre, 'Recall':rec, 'Threshold':threshold, 'NoOfTestSamples':NoOfTestSamples}

import pandas as pd

df = pd.DataFrame(Dict)

print( "\n No. of Gaussians=",M,"No. of cepstral co-efficients=",D)
print( df)
print( "Mean Accuracy = ", np.mean(rec))
df.to_csv('Downloads/librispeech_data/librispeech_results_.csv')
print( "Mean Precision = ", np.mean(pre))
print(filename)

print("---Time Taken to run = %s seconds ---" % (time.time() - script_start_time))


'''   # 24G, 24ceps
Mean Accuracy =  0.726405833673
Mean Precision =  0.998973481608

'''

