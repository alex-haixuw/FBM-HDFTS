import numpy as np 
import matplotlib.pyplot as plt
from dependencies_shuffle import *
from matplotlib import cm
import warnings
import matplotlib as mpl
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

pred_interval = 1
updateind = 9


targetregion = 12
temp1 = np.expand_dims(fD_input[0:(numofobs-testlength-pred_interval+updateind),:,:],2)
temp2 = np.expand_dims(fD_input[pred_interval:(numofobs - testlength+updateind),:,targetregion],2)
temp1 = np.squeeze(temp1,2)
result = Bbasis.fit([temp1,temp2],np.linspace(0,1,numofages),smoothness_constraint = 1, roughness_control = 1e-3, sparsity = 2e-3, niter = 6,numofpredictor=numofregion)
coefmat = result[2]
result[4]



regionlist = [ "Hokkaido",  "Aomori" ,   "Iwate"  ,   "Miyagi"  ,  "Akita"  ,   "Yamagata",  "Fukushima" ,"Ibaraki" ,  "Tochigi",   "Gunma" ,    "Saitama" ,  "Chiba" ,    "Tokyo"  ,   "Kanagawa" ,
"Niigata" ,  "Toyama"  , "Ishikawa"  ,"Fukui"   ,  "Yamanashi" ,"Nagano"  ,  "Gifu" ,     "Shizuoka"  ,"Aichi"    , "Mie"     ,  "Shiga" ,    "Kyoto",     "Osaka"  ,   "Hyogo",     "Nara"  ,   
"Wakayama" , "Tottori"  , "Shimane"  , "Okayama"   ,"Hiroshima" ,"Yamaguchi" ,"Tokushima" ,"Kagawa" ,   "Ehime"   ,  "Kochi" ,    "Fukuoka",   "Saga"  ,    "Nagasaki"  ,"Kumamoto" , "Oita"   ,  
"Miyazaki" , "Kagoshima" ,"Okinawa" ]


colorlist = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr']

u_grid = np.linspace(0,1,numofages)
v_grid = np.linspace(0,1,numofages) #intpoints
U,V = np.meshgrid(v_grid, u_grid)
positions = np.vstack([V.ravel(), U.ravel()]).T

targetregion = 12
temp1 = np.expand_dims(fD_input[0:(numofobs-testlength-pred_interval+updateind),:,:],2)
temp2 = np.expand_dims(fD_input[pred_interval:(numofobs - testlength+updateind),:,targetregion],2)
temp1 = np.squeeze(temp1,2)
result = Bbasis.fit([temp1,temp2],np.linspace(0,1,numofages),smoothness_constraint = 0, roughness_control = 5e-2, sparsity = 2e-3, niter = 6,numofpredictor=numofregion)
coefmat = result[2]


tempvec = np.zeros( (coefmat[:,:,1].shape[0],coefmat[:,:,1].shape[1],3))
for kk in range(3):
    tempvec[:,:,kk] = coefmat[:,:,result[4][kk]]   
#tempvec[np.abs(tempvec) < 5.5e-2] = 0

fig, ax = plt.subplots(2, 3,figsize=(12,8),sharex=True,constrained_layout=True)
ax[0,0].set_xlim([-5,105])
for regulator in range(3): 
    surfacecoef = (Bbasis.data_to_basismat(positions) @ tempvec[:,:,regulator].reshape(-1)).reshape(V.shape)
    surfacecoef[np.abs(surfacecoef) < 1e-2] = 0
    surfacecoef[surfacecoef == 0] = np.nan
    contour = ax[0,regulator].contourf(V*100,U*100,surfacecoef,  cmap=mpl.cm.RdBu)
    ax[0,regulator].set_title(regionlist[result[4][regulator]], fontsize=12, weight='bold') 
    for tt in range(numofyears):
        ax[1,regulator].plot(jp_female_input[tt,:,result[4][regulator]]/2.303, label=str(1973+tt),color =  mpl.colormaps['jet']((tt+1)/numofyears))
        #ax[1,regulator].plot(jp_female_input[tt,:,12]/2.303, label=str(1973+tt),color =  mpl.colormaps['jet']((tt+1)/numofyears))
cbar = fig.colorbar(contour, ax=ax, orientation='vertical', fraction=0.02, pad=0.01,shrink = 0.5,extend = 'both')
#cbar.set_label('', fontsize=12, weight='bold')
cbar.minorticks_on()
ax[0,0].set_ylabel('u', fontsize=12, weight='bold')
ax[1,0].set_ylabel('X(v)', fontsize=12, weight='bold')
fig.supxlabel('v', fontsize=16, weight='bold')
fig.savefig('surface_coef.pdf',bbox_inches='tight')

