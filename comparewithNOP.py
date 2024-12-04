import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd 
jp_female_input = np.load('jp_female_input.npy')/2.303

numofyears  = jp_female_input.shape[0]
numofages   = jp_female_input.shape[1]
numofregion = jp_female_input.shape[2]
numofobs = numofyears


abserror_UFTS = np.load('UFTS_MAE.npy')
abserror_MFTS = np.load('MFTS_MAE.npy')
abserror_MLFTS = np.load('MLFTS_MAE.npy')
abserror_NOP = np.zeros(abserror_UFTS.shape)
abserror_FBM = np.zeros(abserror_UFTS.shape)



rmse_UFTS = np.load('UFTS_RMSE.npy')
rmse_MFTS = np.load('MFTS_RMSE.npy')
rmse_MLFTS = np.load('MLFTS_RMSE.npy')
rmse_NOP = np.zeros(rmse_UFTS.shape)
rmse_FBM = np.zeros(rmse_UFTS.shape)

regionlist = [ "Hokkaido",  "Aomori" ,   "Iwate"  ,   "Miyagi"  ,  "Akita"  ,   "Yamagata",  "Fukushima" ,"Ibaraki" ,  "Tochigi",   "Gunma" ,    "Saitama" ,  "Chiba" ,    "Tokyo"  ,   "Kanagawa" ,
"Niigata" ,  "Toyama"  , "Ishikawa"  ,"Fukui"   ,  "Yamanashi" ,"Nagano"  ,  "Gifu" ,     "Shizuoka"  ,"Aichi"    , "Mie"     ,  "Shiga" ,    "Kyoto",     "Osaka"  ,   "Hyogo",     "Nara"  ,   
"Wakayama" , "Tottori"  , "Shimane"  , "Okayama"   ,"Hiroshima" ,"Yamaguchi" ,"Tokushima" ,"Kagawa" ,   "Ehime"   ,  "Kochi" ,    "Fukuoka",   "Saga"  ,    "Nagasaki"  ,"Kumamoto" , "Oita"   ,  
"Miyazaki" , "Kagoshima" ,"Okinawa" ]




'''
region_id = [12,5,39]


colorlist = ['Blues', 'Oranges']

fig, ax = plt.subplots(2,2, figsize=(8,6),sharex=True, sharey=True,constrained_layout=True)

for tt in range(testlength):
    ax[0,0].plot(np.arange(numofages),FBM_predmat[tt,:,12], label='FBM',color= mpl.colormaps['jet']((tt+1)/testlength),linewidth=2)
    ax[0,1].plot(np.arange(numofages),testmat[tt,:,12], label='True',color=mpl.colormaps['jet']((tt+1)/testlength),linewidth = 2)
    ax[0,1].text(20,-0.5, str(regionlist[12]), fontsize=12,weight='bold')

for tt in range(testlength):
    ax[1,0].plot(np.arange(numofages),FBM_predmat[tt,:,5], label='FBM',color= mpl.colormaps['jet']((tt+1)/testlength),linewidth=2)
    ax[1,1].plot(np.arange(numofages),testmat[tt,:,5], label='True',color=mpl.colormaps['jet']((tt+1)/testlength),linewidth = 2)
    ax[1,1].text(20,-0.5, str(regionlist[5]), fontsize=12,weight='bold')
        
fig.savefig('predictionresult.pdf', bbox_inches='tight')
'''

#Validation for 10 years 
for pred_interval in range(1,11):
    testlength = 11 - pred_interval 
    NOP_predmat = np.load('./Data/NOP_pred_step'+str(pred_interval)+'.npy')
    FBM_predmat = np.load('./Data/FBM_pred_step'+str(pred_interval)+'.npy')
    
    testmat = jp_female_input[-testlength:,:,:]
    abserror_NOP[:,pred_interval-1,:,0] = np.mean(np.abs(NOP_predmat - testmat),0).T
    abserror_FBM[:,pred_interval-1,:,0] = np.mean(np.abs(FBM_predmat - testmat),0).T

    rmse_NOP[:,pred_interval-1,:,0] = np.sqrt(np.mean((NOP_predmat - testmat)**2,0).T)
    rmse_FBM[:,pred_interval-1,:,0] = np.sqrt(np.mean((FBM_predmat - testmat)**2,0).T)

comparedata = np.zeros((numofregion, 10,numofages, 5))


comparedata[:,:,:,0] = abserror_UFTS[:,:,:,0]
comparedata[:,:,:,1] = abserror_MFTS[:,:,:,0]
comparedata[:,:,:,2] = abserror_MLFTS[:,:,:,0]
comparedata[:,:,:,3] = abserror_NOP[:,:,:,0]
comparedata[:,:,:,4] = abserror_FBM[:,:,:,0]
np.mean(comparedata[:,:,:],(0,2))


#2.

comparedata = np.zeros((numofregion, 10,numofages, 5))
comparedata[:,:,:,0] = rmse_UFTS[:,:,:,0]
comparedata[:,:,:,1] = rmse_MFTS[:,:,:,0]
comparedata[:,:,:,2] = rmse_MLFTS[:,:,:,0]
comparedata[:,:,:,3] = rmse_NOP[:,:,:,0]
comparedata[:,:,:,4] = rmse_FBM[:,:,:,0]
np.mean(comparedata[:,:,:],(0,2))


boxplotdata = np.mean(comparedata[:,np.array([0,4,9]),:,:],2)[:,:,np.array([4,3,0,1,2])]

time_lags = [1, 5, 10]
methods = ['FBM','NOP','UFTS','MFTS','MLFTS']
data = []

for i, lag in enumerate(time_lags):
    for j, method in enumerate(methods):
        for value in boxplotdata[:, i, j]:
            data.append([lag, method, value])

# Convert to a DataFrame
df = pd.DataFrame(data, columns=['Time Lag', 'Method', 'MAFE'])
# Create the violin plot
plt.figure(figsize=(12, 5))
sns.violinplot(x='Time Lag', y='MAFE', hue='Method', data=df)
plt.title(r'Distributions of MAFE for different prefectures of Japan with $\delta = 1, 5, 10$', fontsize=15, fontweight='bold')
plt.xlabel('Time Lag', fontsize=15, fontweight='bold')
plt.ylabel('MAFE', fontsize=15, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.savefig('violin_plot_comparewithNOP.pdf', bbox_inches='tight')
plt.show()




#3.

fig, ax = plt.subplots(figsize = (8,6),constrained_layout=True)
ax.plot(np.arange(numofages), np.mean(comparedata, (0,1 ))[:,0],label='UFTS',linewidth=2)
ax.plot(np.arange(numofages),np.mean(comparedata, (0,1 ))[:,1],label='MFTS',linewidth=2)
ax.plot(np.arange(numofages),np.mean(comparedata, (0,1 ))[:,2],label='MLFTS',linewidth=2)
ax.plot(np.arange(numofages),np.mean(comparedata, (0,1 ))[:,3],label='NOP',linewidth=2)
ax.plot(np.arange(numofages),np.mean(comparedata, (0,1 ))[:,4],label='FBM',linewidth=2)
fig.legend(loc='upper center', ncol=5, fancybox=True, shadow=True,prop={'weight': 'bold'})
fig.savefig('comparewithNOP.pdf', bbox_inches='tight')




fig, ax = plt.subplots(2,2, figsize=(10,10))
axes = ax.ravel()
regionid = 0
pred_interval = 1
gender_ind = 0
axes[0].plot(abserror_UFTS[regionid, pred_interval-1,:,gender_ind], label='UFTS')
axes[0].plot(abserror_NOP[regionid, pred_interval - 1, :, gender_ind], label='NOP')
axes[0].plot(abserror_FBM[regionid, pred_interval - 1, :, gender_ind], label='FBM')



regionid = 3
pred_interval = 2
gender_ind = 0
axes[1].plot(abserror_UFTS[regionid, pred_interval-1,:,gender_ind], label='UFTS')
axes[1].plot(abserror_NOP[regionid, pred_interval - 1, :, gender_ind], label='NOP')
axes[1].plot(abserror_FBM[regionid, pred_interval - 1, :, gender_ind], label='FBM')


regionid = 10
pred_interval = 5
gender_ind = 0
axes[2].plot(abserror_UFTS[regionid, pred_interval-1,:,gender_ind], label='UFTS')
axes[2].plot(abserror_NOP[regionid, pred_interval - 1, :, gender_ind], label='NOP')
axes[2].plot(abserror_FBM[regionid, pred_interval - 1, :, gender_ind], label='FBM')

regionid = 20
pred_interval = 4
gender_ind = 0
axes[3].plot(abserror_UFTS[regionid, pred_interval-1,:,gender_ind], label='UFTS')
axes[3].plot(abserror_NOP[regionid, pred_interval - 1, :, gender_ind], label='NOP')
axes[3].plot(abserror_FBM[regionid, pred_interval - 1, :, gender_ind], label='FBM')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.savefig('previewresult.pdf', bbox_inches='tight')



regionid = 5
forecastid = 2
plt.plot(testmat[forecastid, :, regionid], label='True')
plt.plot(NOP_predmat[forecastid, :, regionid], label='NOP')
plt.plot(FBM_predmat[forecastid, :, regionid], label='FBM')
plt.legend()
plt.show()
regionid = 30
plt.plot(np.mean(abserror_NOP, 0)[:,regionid], label='NOP')
plt.plot(np.mean(abserror_FBM, 0)[:,regionid], label='FBM')
plt.legend()
plt.show()



