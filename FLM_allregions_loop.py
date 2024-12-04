import numpy as np 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from math import factorial
from functools import partial
import plotly.graph_objects as go
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D
from dependencies import *
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from matplotlib import cm, ticker
from sklearn import linear_model
import scipy 
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

jp_female_input = np.load('jp_female_input.npy')
#Leave the last 10 years as the test set 
#jp_female_input = jp_female_input[0:40,:,:]
#First order differencing of the time series 
fD_input = np.diff(jp_female_input,axis=0)

#Validation for 10 years 
testlength = 10

numofyears  = fD_input.shape[0]
numofages   = fD_input.shape[1]
numofregion = fD_input.shape[2]
numofobs = numofyears


#Do this for all regions at once 


xslice = 6
yslice = 4


x_partition = np.unique(np.concatenate([np.linspace(0, .2, xslice) , np.linspace(0.2, 1, 3)]).ravel())
y_partition = np.unique(np.concatenate([np.linspace(0, .2, yslice) , np.linspace(0.2, 1, 3)]).ravel())

tri_vertices = np.vstack((np.repeat(x_partition, yslice +2), np.tile(y_partition, xslice + 2))).T
Bbasis  = triBpoly(2, tri_vertices)


#One perfecture at a time 
for pred_interval in tqdm(range(1,11)):
    matrixname = './testresult/female_multiregion_pred_'+str(pred_interval)+'ahead.npy'
    temp = np.zeros((11-pred_interval, int(numofregion), numofages))
    for targetregion in tqdm(range(numofregion)):
        print(str(pred_interval)+'-steapahead')
        print('Region: '+str(targetregion))
        for updateind in tqdm(range(0,11-pred_interval)):
            temp1 = np.expand_dims(fD_input[0:(numofobs-testlength-pred_interval+updateind),:,:],2)
            temp2 = np.expand_dims(fD_input[pred_interval:(numofobs - testlength+updateind),:,targetregion],2)
            temp1 = np.squeeze(temp1,2)
            Y_vec = temp2.reshape(-1)
            result = Bbasis.fit([temp1,temp2],np.linspace(0,1,numofages),
                smoothness_constraint = 0, roughness_control = 1e-2,
                sparsity = 1e-3, niter = 1000,numofpredictor=numofregion)
            lastobs = np.expand_dims(np.squeeze(np.expand_dims(fD_input[pred_interval:(numofobs - testlength+updateind),:,:],2),2)[-1,:,:],0)
            Y_pred = Bbasis.predict(lastobs)        
            temp[updateind,targetregion,:] = Y_pred.ravel()
    np.save(matrixname,temp)


