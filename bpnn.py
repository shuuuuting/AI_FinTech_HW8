#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
#from PIL import Image
import random
import matplotlib.pyplot as plt

npzfile = np.load('CBCL.npz')
trainface = npzfile['arr_0']
trainnonface = npzfile['arr_1']
testface = npzfile['arr_2']
testnonface = npzfile['arr_3']

raw = np.zeros((40*19,50*19))
for y in range(40):
    for x in range(50):
        I1 = trainface[y*50+x,:].reshape((19,19))
        raw[y*19:y*19+19,x*19:x*19+19] = I1

#I = Image.fromarray(raw)
#I.show()

#Back prop
def BPNNtrain(pf,nf,hn,lr,iteration): #pf:trainface接近1/nf:nonface接近0/hn:hidden node/lr:learning rate/data要反覆學iteration次
    pn = pf.shape[0] #positive sample
    nn = nf.shape[0] #negative sample
    fn = pf.shape[1] 
    feature = np.append(pf,nf,axis=0)
    target = np.append(np.ones((pn,1)),np.zeros((nn,1)),axis=0)
    WI = np.random.normal(0,1,(fn+1,hn)) #input weight
    WO = np.random.normal(0,1,(hn+1,1)) #output weight
    for t in range(iteration):
        s = random.sample(range(pn+nn),pn+nn)
        for i in range(pn+nn):
            ins = np.append(feature[s[i],:],1) #加1當常數項
            oh = ins.dot(WI) #hidden layer output
            oh = 1/(1+np.exp(-oh)) #sigmoid func.
            hs = np.append(oh,1) #hidden node signal
            out = hs.dot(WO) #output node signal(answer)
            out = 1/(1+np.exp(-out))
            dk = out*(1-out)*(target[s[i]]-out) #delta k
            dh = oh*(1-oh)*WO[:hn,0]*dk #delta h 
            WO[:,0] += lr*dk*hs #權重更新
            for j in range(hn):
                WI[:,j] += lr*dh[j]*ins
    model = dict()
    model['WI'] = WI
    model['WO'] = WO
    return model

def BPNNtest(feature,model):
    sn = feature.shape[0]
    WI = model['WI']
    WO = model['WO']
    out = np.zeros((sn,1))
    for i in range(sn):
        ins = np.append(feature[i,:],1) #加1當常數項
        oh = ins.dot(WI) #hidden layer output
        oh = 1/(1+np.exp(-oh)) #sigmoid func.
        hs = np.append(oh,1) #hidden node signal
        out[i] = hs.dot(WO) #output node signal(answer)
        out[i] = 1/(1+np.exp(-out[i])) #越接近1代表越像人臉
    return out

network = BPNNtrain(trainface/255,trainnonface/255,20,0.01,10)
pscore = BPNNtest(trainface/255,network)
nscore = BPNNtest(trainnonface/255,network)

#ROC curve
X = np.zeros(99)
Y = np.zeros(99)
for i in range(99):
    threshold = (i+1)/100
    X[i] = np.mean(nscore>threshold)
    Y[i] = np.mean(pscore>threshold)
plt.plot(X,Y)