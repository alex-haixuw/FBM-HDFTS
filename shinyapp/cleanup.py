import numpy as np 


FBM_RMSE = np.load('./data/FBM_RMSE.npy')[:,:,:,0]
FBM_MAE = np.load('./data/FBM_MAE.npy')[:,:,:,0]

NOP_RMSE = np.load('./data/NOP_RMSE.npy')[:,:,:,0]
NOP_MAE = np.load('./data/NOP_MAE.npy')[:,:,:,0]


UFTS_RMSE = np.load('./data/UFTS_RMSE.npy')[:,:,:,0]
UFTS_MAE = np.load('./data/UFTS_MAE.npy')[:,:,:,0]


MLFTS_RMSE = np.load('./data/MLFTS_RMSE.npy')[:,:,:,0]
MLFTS_MAE = np.load('./data/MLFTS_MAE.npy')[:,:,:,0]

MFTS_RMSE = np.load('./data/MFTS_RMSE.npy')[:,:,:,0]
MFTS_MAE = np.load('./data/MFTS_MAE.npy')[:,:,:,0]


RMSEmat = np.zeros((47,10,101,5))
MAEmat  = np.zeros((47,10,101,5))

RMSEmat[:,:,:,0] = FBM_RMSE
RMSEmat[:,:,:,1] = NOP_RMSE
RMSEmat[:,:,:,2] = UFTS_RMSE
RMSEmat[:,:,:,3] = MLFTS_RMSE
RMSEmat[:,:,:,4] = MFTS_RMSE

MAEmat[:,:,:,0] = FBM_MAE
MAEmat[:,:,:,1] = NOP_MAE
MAEmat[:,:,:,2] = UFTS_MAE
MAEmat[:,:,:,3] = MLFTS_MAE
MAEmat[:,:,:,4] = MFTS_MAE

np.save('./data/RMSEmat.npy', RMSEmat)
np.save('./data/MAEmat.npy', MAEmat)