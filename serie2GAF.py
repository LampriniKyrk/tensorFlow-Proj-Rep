# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:17:18 2014
Modified on Wed Jan 27 15:36:00 2016
@author: Stephen Wang
"""

from serie2QMlib import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cPickle, sys

#Define sliding window
def window_time_series(series, n, step = 1):
#    print "in window_time_series",series
    if step < 1.0:
        step = max(int(step * n), 1)
    return [series[i:i+n] for i in range(0, len(series) - n + 1, step)]

#PAA function
def paa(series, now, opw):
    if now == None:
        now = len(series) / opw
    if opw == None:
        opw = len(series) / now
    return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]

def standardize(serie):
    dev = np.sqrt(np.var(serie))
    mean = np.mean(serie)
    return [(each-mean)/dev for each in serie]

#Rescale data into [0,1]
def rescale(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap for each in serie]
#Rescale data into [-1,1]    
def rescaleminus(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap*2-1 for each in serie]

#Generate quantile bins
def QMeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    MSM = np.zeros([Q,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)-1):
        MSM[label[i]][label[i+1]] += 1
    for i in xrange(Q):
        if sum(MSM[i][:]) == 0:
            continue
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM), label, q.levels

#Generate quantile bins when equal values exist in the array (slower than QMeq)
def QVeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    qv = np.zeros([1,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0,len(label)):
        qv[0][label[i]] += 1.0        
    return np.array(qv[0][:]/sum(qv[0][:])), label

#Generate Markov Matrix given a spesicif number of quantile bins
def paaMarkovMatrix(paalist,level):
    paaindex = []
    for each in paalist:    
        for k in range(len(level)):
            lower = float(level[k][1:-1].split(',')[0])
            upper = float(level[k][1:-1].split(',')[-1])
            if each >=lower and each <= upper:
                paaindex.append(k)
    return paaindex

# Generate pdf files of generated images
def gengrampdfs(image,paaimages,label,name):
    import matplotlib.backends.backend_pdf as bpdf
    import operator
    index = zip(range(len(label)),label)
    index.sort(key = operator.itemgetter(1))
    with bpdf.PdfPages(name) as pdf:
        count = 0
        for p,q in index:
            count += 1
            print 'generate fig of pdfs:',p
            plt.ioff();fig= plt.figure();plt.suptitle(datafile+'_'+str(label[p]));ax1 = plt.subplot(121);plt.imshow(image[p]);divider = make_axes_locatable(ax1);cax = divider.append_axes("right", size="5%", pad=0.1);plt.colorbar(cax = cax);ax2 = plt.subplot(122);plt.imshow(paaimage[p]);divider = make_axes_locatable(ax2);cax = divider.append_axes("right", size="5%", pad=0.1);plt.colorbar(cax = cax);
            pdf.savefig(fig)
            plt.savefig("fig.png")
            plt.close(fig)
            if count > 3:
                break
    pdf.close

# Generate pdf files of trainsisted array in porlar coordinates
def genpolarpdfs(raw,label,name):
    import matplotlib.backends.backend_pdf as bpdf
    import operator
    index = zip(range(len(label)),label)
    index.sort(key = operator.itemgetter(1))
    with bpdf.PdfPages(name) as pdf:
        for p,q in index:
            print 'generate fig of pdfs:',p
            plt.ioff();r = np.array(range(1,length+1));r=r/100.0;theta = np.arccos(np.array(rescaleminus(standardize(raw[p][1:]))))*2;fig=plt.figure();plt.suptitle(datafile+'_'+str(label[p]));ax = plt.subplot(111, polar=True);ax.plot(theta, r, color='r', linewidth=3);
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close

#return the max value instead of mean value in PAAs
def maxsample(mat, s):
    retval = []
    x, y, z = mat.shape
    l = np.int(np.floor(y/float(s)))
    for each in mat:
        block = []
        for i in range(s):
            block.append([np.max(each[i*l:(i+1)*l,j*l:(j+1)*l]) for j in xrange(s)])
        retval.append(np.asarray(block))
    return np.asarray(retval)

#Pickle the data and save in the pkl file
def pickledata(mat, label, train, name):
    print '..pickling data:',name
    traintp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    f = file('dishwasher32/'+name+'.pkl', 'wb')
    pickletp = [traintp, testtp]
    cPickle.dump(pickletp, f, protocol=cPickle.HIGHEST_PROTOCOL)

def pickle3data(mat, label, train, name):
    print '..pickling data:',name
    traintp = (mat[:train], label[:train])
    validtp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    f = file(name+'.pkl', 'wb')
    pickletp = [traintp, validtp, testtp]
    cPickle.dump(pickletp, f, protocol=cPickle.HIGHEST_PROTOCOL)
    
#################################
###Define the parameters here####
#################################

datafiles = ['dish washer1'] # Data fine name
trains = [250] # Number of training instances (because we assume training and test data are mixed in one file)
size = [32]#[64]  # PAA size
GAF_type = 'GADF' # GAF type: GASF, GADF
save_PAA = True # Save the GAF with or without dimension reduction by PAA: True, False
rescale_type = 'Zero' # Rescale the data into [0,1] or [-1,1]: Zero, Minusone



for datafile, train in zip(datafiles,trains):


    fn = datafile
    for s in size:  
        print 'read file', datafile, 'size',s, 'GAF type', GAF_type
        raw = open(fn).readlines()
        raw = [map(float, each.strip().split()) for each in raw]

        labelLines = open('dish washer1-labels').readlines()
        labelLines = [map(float, labelline.strip().split()) for labelline in labelLines]
        length = len(raw[0])-1
        
        print 'format data'
        # label = []
        # image = []
        # paaimage = []
        # patchimage = []
        # matmatrix = []
        # fullmatrix = []
        i=0
        for each, labelline in zip(raw, labelLines):
            label = []
            image = []
            paaimage = []
            patchimage = []
            matmatrix = []
            fullmatrix = []

            label.append(labelline)#(np.average(labelline))
            # if rescale_type == 'Zero':
            #     std_data = rescale(each[1:])
            # elif rescale_type == 'Minusone':
            #     std_data = rescaleminus(each[1:])
            # else:
            #     sys.exit('Unknown rescaling type!')
            std_data = each
            paalistcos = paa(std_data,s,None) 
            #paalistcos = rescale(paa(each[1:],s,None)) 
            #paalistcos = rescaleminus(paa(each[1:],s,None))
            
            ################raw###################                
            datacos = np.array(std_data)
            datasin = np.sqrt(1-np.array(std_data)**2)

            paalistcos = np.array(paalistcos)
            paalistsin = np.sqrt(1-paalistcos**2)
            
            datacos = np.matrix(datacos)
            datasin = np.matrix(datasin)            
            
            paalistcos = np.matrix(paalistcos)
            paalistsin = np.matrix(paalistsin)            
            if GAF_type == 'GASF':
                paamatrix = paalistcos.T*paalistcos-paalistsin.T*paalistsin
                matrix = np.array(datacos.T*datacos-datasin.T*datasin)
            elif GAF_type == 'GADF':
                paamatrix = paalistsin.T*paalistcos-paalistcos.T*paalistsin
                matrix = np.array(datasin.T*datacos - datacos.T*datasin)
            else:
                sys.exit('Unknown GAF type!')
            paamatrix = np.array(paamatrix)
            image.append(matrix)
            paaimage.append(np.array(paamatrix))
            matmatrix.append(paamatrix.flatten())
            fullmatrix.append(matrix.flatten())

            label = np.asarray(label)
            image = np.asarray(image)
            paaimage = np.asarray(paaimage)
            patchimage = np.asarray(patchimage)
            matmatrix = np.asarray(matmatrix)
            fullmatrix = np.asarray(fullmatrix)
            # maximage = maxsample(image, s)
            # maxmatrix = np.asarray(np.asarray([each.flatten() for each in maximage]))

            datafilename = datafile + '_PAA_' + str(s) + '_' + GAF_type +' '+str(i)
            if save_PAA == False:
                finalmatrix = matmatrix
            else:
                finalmatrix = fullmatrix
            pickledata(finalmatrix, label, train, datafilename)
            i += 1
            # gengrampdfs(image,paaimage,label,'samplePdf'+str(i))
            # if i>1000:
            #     break

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #~~~~~ ORIGINAL CODE ~~~~~~~~~~~~
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # label = np.asarray(label)
        # image = np.asarray(image)
        # paaimage = np.asarray(paaimage)
        # patchimage = np.asarray(patchimage)
        # matmatrix = np.asarray(matmatrix)
        # fullmatrix = np.asarray(fullmatrix)
        # #maximage = maxsample(image, s)
        # #maxmatrix = np.asarray(np.asarray([each.flatten() for each in maximage]))
        #
        # datafilename = datafile +'_PAA_'+str(s)+'_'+ GAF_type
        # if save_PAA == False:
        #     finalmatrix = matmatrix
        # else:
        #     finalmatrix = fullmatrix
        # pickledata(finalmatrix, label, train, datafilename)
#        writeQM2PretrainMat(datafilename, finalmatrix.T,1.0)
#        writeQM2Mat(datafilename+'_train', finalmatrix[:train].T,label[:train])
#        writeQM2Mat(datafilename+'_test', finalmatrix[train:].T,label[train:])
##
    
#        datafilename = datafile +'_gram_'+'full_1drcos-1'
#        writeQM2PretrainMat(datafilename, fullmatrix.T,1.0)
#        writeQM2Mat(datafilename+'_train', fullmatrix[:train].T,label[:train])
#        writeQM2Mat(datafilename+'_test', fullmatrix[train:].T,label[train:])

# polar coordinates
#k=0;r = np.array(range(1,length+1));r=r/100.0;theta = np.array(rescale(raw[k][1:]))*2*np.pi;plt.figure();ax = plt.subplot(111, polar=True);ax.plot(theta, r, color='r', linewidth=3);plt.show()
## draw large image and paa image
k = 0;plt.figure();plt.suptitle(datafile+'_index_'+str(k)+'_label_'+str(label[k]));ax1 = plt.subplot(121);plt.title(GAF_type + 'without PAA');plt.imshow(image[k]);divider = make_axes_locatable(ax1);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);ax2 = plt.subplot(122);plt.title(GAF_type + 'with PAA');plt.imshow(paaimage[k]);divider = make_axes_locatable(ax2);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);
#plt.figure();plt.suptitle(datafile+'_index_'+str(k)+'_label_'+str(label[k]));ax1 = plt.subplot(131);plt.imshow(image[k]);divider = make_axes_locatable(ax1);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);ax2 = plt.subplot(132);plt.imshow(paaimage[k]);divider = make_axes_locatable(ax2);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);ax3 = plt.subplot(133);plt.imshow(patchimage[k]);divider = make_axes_locatable(ax3);cax = divider.append_axes("right", size="5%", pad=0.2);plt.colorbar(cax = cax);
#datafile+'_'+str(label[k])+'_'+str(Q)
