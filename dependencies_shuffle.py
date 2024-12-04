import numpy as np 
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from math import factorial
from functools import partial
import math
from scipy.linalg import block_diag
import plotly.graph_objects as go
from sklearn import linear_model

class triBpoly:
    def __init__(self, degree:int,  tri_vertices: np.ndarray):
        self.degree = degree
        temp = np.array(np.meshgrid(np.arange(self.degree+1),np.arange(self.degree+1),np.arange(self.degree+1))).T.reshape(-1,3)
        temp = temp[temp.sum(axis=1) == self.degree,:]
        DgrComb = temp[temp[:, 0].argsort()[::-1]]
        DgrComb[:, [1, 2]] = DgrComb[:, [2, 1]]
        self.DgrComb = DgrComb
        self.nbasispertri = self.DgrComb.shape[0]
        self.tri = Delaunay(tri_vertices)
        self.nTriangles = self.tri.nsimplex
        self.totalbasisnum = self.nbasispertri * self.nTriangles
        self.tribasisindex = np.arange(0,self.nTriangles * self.nbasispertri).reshape(self.nTriangles, self.nbasispertri)
        
            
    def ijk_comb(self, degree:int):
        temp = np.array(np.meshgrid(np.arange(degree+1),np.arange(degree+1),np.arange(degree+1))).T.reshape(-1,3)
        temp = temp[temp.sum(axis=1) == degree,:]
        DgrComb = temp[temp[:, 0].argsort()[::-1]]
        DgrComb[:, [1, 2]] = DgrComb[:, [2, 1]]
        return DgrComb


    def convert_to_barycentric(self, xypts: np.ndarray):
        triindex = self.tri.find_simplex(xypts)
        barcoord = np.empty((xypts.shape[0],3))
        for jj in range(triindex.shape[0]):
            b =  self.tri.transform[triindex[jj],:2].dot(np.transpose(xypts[np.array([jj]),:] -  self.tri.transform[triindex[jj],2]))
            barcoord[jj,:] = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
        return barcoord, triindex
    
    
    def convert_to_euclidean(self, bpts: np.ndarray, triindex: np.ndarray):
        Euc_coord = np.zeros((bpts.shape[0],2))
        for jj in range(triindex.shape[0]):
            Euc_coord[jj,:] = self.tri.points[self.tri.simplices[triindex[jj]],:].T @ bpts[jj,:].T
        return Euc_coord
    
    
    def Bpoly(self,barcoorvec,i:int,j:int,k:int):
        return factorial(i+j+k) / (factorial(i) * factorial(j) * factorial(k)) * barcoorvec[0]**i * barcoorvec[1]**j * barcoorvec[2]**k

    def eval(self, barcoord, DgrComb):
        basismat = np.zeros((barcoord.shape[0], DgrComb.shape[0]))
        for jj in range(DgrComb.shape[0]):
            for kk in range(barcoord.shape[0]):
                basismat[kk,jj] = self.Bpoly(barcoord[kk,:], DgrComb[jj,0], DgrComb[jj,1], DgrComb[jj,2])
        return basismat
    
    def data_to_basismat(self, xydata:np.ndarray):
        barcoord, triindex = self.convert_to_barycentric(xydata)
        self.triindex = triindex
        basisvalue = self.eval(barcoord,self.DgrComb)
        storeind = np.arange(0,self.nTriangles * self.nbasispertri).reshape(self.nTriangles, self.nbasispertri)
        bardata = np.zeros((xydata.shape[0], self.nTriangles * self.nbasispertri))
        for kk in range(xydata.shape[0]):
            bardata[kk,storeind[triindex[kk],:]] = basisvalue[kk,:]
        return bardata

    def Hmat(self, orderR:int):
        
        degree = self.degree #degree = 3
        
        #H mat  number of unique shared edges * number of basis per triangle 
        neighboringlist  = []
        for kk in range(self.nTriangles):
            numnei = np.where(self.tri.neighbors[kk] != -1)[0]
            neighboringlist.append(np.c_[np.repeat(kk, numnei.shape[0]),self.tri.neighbors[kk][np.where(self.tri.neighbors[kk] != -1)]])
            
        neighborinTriangluation = np.concatenate(neighboringlist,axis = 0)

        for kk in range(neighborinTriangluation.shape[0]):
            neighborinTriangluation[kk,:] = np.sort(neighborinTriangluation[kk,:])
                
        sharededgelist = np.unique(neighborinTriangluation, axis = 0)

        self.sharededgelist = sharededgelist

        Hmat = np.zeros((sharededgelist.shape[0] * self.nbasispertri, self.nTriangles * self.nbasispertri))


        #hard code the indices for now , degree = 1, 2, 3

        rightblock_index = np.array([0,0,1, 2,2, 1, 3, 5,4, 4,5, 3 , 6, 9, 7,8,8,7,9,6]).reshape(10,2) 
        Hmat_rightblock = np.zeros((int((degree+1) * (degree+2) * 0.5), self.nbasispertri))
        take_index = rightblock_index[0:np.sum(np.arange(1,(degree+2))),:]
        Hmat_rightblock[take_index[:,0],take_index[:,1]] = -1



        #Each row block is a linear system of coefficients
        Hmat_rowind   = np.arange(sharededgelist.shape[0] * self.nbasispertri).reshape(sharededgelist.shape[0],self.nbasispertri)
        #The column block is the set of coefficients for each triangle
        Tri_coefind   = np.arange(self.nTriangles * self.nbasispertri).reshape(self.nTriangles, self.nbasispertri)
        #Smoothness constraints for different l from largest to smallest 
        withinrow_ind = np.split(np.arange(0,self.nbasispertri),np.cumsum(np.arange(1,self.degree + 2)).tolist())[:-1]


        #hard code filling indices? degree = 3
        #fillind = [[np.arange(0,self.nbasispertri).tolist()], [np.array([1,4,5,6,7,8]),np.array([2,4,5,7,8,9])], [[3,6,7],[4,7,8],[5,8,9]],[[6],[7],[8],[9]]]
        fillind = [[np.arange(0,self.nbasispertri).tolist()],[np.array([1,3,4]),np.array([2,4,5])],[[3],[4],[5]]]

        for ii in range(sharededgelist.shape[0]):
            
            tri1 = sharededgelist[ii,0]
            tri2 = sharededgelist[ii,1]
            v4 = self.tri.points[np.setdiff1d(self.tri.simplices[tri2,:], self.tri.simplices[tri1,:]),:]
            
            
            b = self.tri.transform[tri1,:2].dot(np.transpose(v4 -  self.tri.transform[tri1,2]))
            v4_bary = np.c_[np.transpose(b), 1 - b.sum(axis=0)]
            
            
            Hmat_leftblock = np.zeros((int((degree+1) * (degree+2) * 0.5), self.nbasispertri))
            
            deg_itind = degree
            for kk in range(len(withinrow_ind)):
                for row in range(withinrow_ind[kk].shape[0]):
                    Hmat_leftblock[withinrow_ind[kk][row],fillind[kk][row]] = self.eval(barcoord=v4_bary, DgrComb=self.ijk_comb(deg_itind))
                deg_itind -= 1
            
            
            Hmat[Hmat_rowind[ii,:].min():(Hmat_rowind[ii,:].max()+1),Tri_coefind[tri1,:].min():(Tri_coefind[tri1,:].max()+1)] = Hmat_leftblock
            Hmat[Hmat_rowind[ii,:].min():(Hmat_rowind[ii,:].max()+1),Tri_coefind[tri2,:].min():(Tri_coefind[tri2,:].max()+1)] = Hmat_rightblock        

        #Based on order R, choose rows of the H matrix 
        tochoose = withinrow_ind[::-1][0:(orderR+1)][::-1]

        Hmat_rowind_chosen = Hmat_rowind[:,sum([ kk.tolist() for kk in tochoose],[])].reshape(-1)
        return Hmat[Hmat_rowind_chosen,:]
    
    def get_area(self, x, y):
        area = 0.5 * (x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2]
                  * (y[0] - y[1]))
        return area

    def decasteljau_Derivative_Cmat(self, triangle_vertices, direction, degree = None):
        if degree == None:
            degree = self.degree
        triarea = self.get_area(triangle_vertices[:,0], triangle_vertices[:,1])
        if direction == 'x':
            directional_vec_x = np.array([triangle_vertices[1,1]-triangle_vertices[2,1],
                                        triangle_vertices[2,1]-triangle_vertices[0,1],
                                        triangle_vertices[0,1]-triangle_vertices[1,1]]) / (2 * triarea)
            directional_vec_y = directional_vec_x
        if direction == 'y':
            directional_vec_y = np.array([triangle_vertices[2,0]-triangle_vertices[1,0],
                                        triangle_vertices[0,0]-triangle_vertices[2,0],
                                        triangle_vertices[1,0]-triangle_vertices[0,0]])/ (2 * triarea)
            directional_vec_x = directional_vec_y
        if direction == 'yx':
            directional_vec_x = np.array([triangle_vertices[1,1]-triangle_vertices[2,1],
                                        triangle_vertices[2,1]-triangle_vertices[0,1],
                                        triangle_vertices[0,1]-triangle_vertices[1,1]]) / (2 * triarea)
            directional_vec_y = np.array([triangle_vertices[2,0]-triangle_vertices[1,0],
                                        triangle_vertices[0,0]-triangle_vertices[2,0],
                                        triangle_vertices[1,0]-triangle_vertices[0,0]])/ ( 2 * triarea)
        
        self.crossprod = directional_vec_x.reshape(3,1) @ directional_vec_y.reshape(3,1).T
        crossprod = self.crossprod
        downgradelist = [[[0,0]],[[0,1],[1,0]],[[0,2],[2,0]],[[1,1]],[[1,2],[2,1]],[[2,2]]]

        Cmat = np.zeros((self.nbasispertri, self.ijk_comb(self.degree - 2).shape[0]))
        for kk in range(self.nbasispertri):
            tempa = self.DgrComb[kk,:] - self.ijk_comb(2) 
            downgrade_comb = tempa[np.all(tempa>=0,1),:]
            coefind = np.where(np.all(tempa>=0,1))[0]
            for downcomb in range(downgrade_comb.shape[0]):
                fillind = np.where((self.ijk_comb(self.degree - 2) == downgrade_comb[downcomb]).all(axis=1))[0][0] #find matching rows 
                if len(downgradelist[coefind[downcomb]]) == 2:
                    Cmat[kk,fillind] = crossprod[downgradelist[coefind[downcomb]][0][0],downgradelist[coefind[downcomb]][0][1]] + crossprod[downgradelist[coefind[downcomb]][1][0],downgradelist[coefind[downcomb]][1][1]]
                elif len(downgradelist[coefind[downcomb]]) == 1:
                    Cmat[kk,fillind] = crossprod[downgradelist[coefind[downcomb]][0][0],downgradelist[coefind[downcomb]][0][1]] 
        Cmat = degree * (degree - 1) * Cmat
        return Cmat

    
    def M_tau(self, triangle_vertices, degree = None):
        
        if degree == None:
            degree = self.degree
            
        resultmat = np.zeros((self.ijk_comb(degree -2).shape[0],self.ijk_comb(degree -2).shape[0]))
        for rowind in range(self.ijk_comb(degree -2).shape[0]):
            for colind in range(self.ijk_comb(degree -2).shape[0]):
                ii       = self.ijk_comb(degree -2)[rowind,0]
                ii_prime = self.ijk_comb(degree -2)[colind,0]
                jj       = self.ijk_comb(degree -2)[rowind,1]
                jj_prime = self.ijk_comb(degree -2)[colind,1]
                kk       = self.ijk_comb(degree -2)[rowind,2]
                kk_prime = self.ijk_comb(degree -2)[colind,2]
                resultmat[rowind, colind] = math.comb(ii + ii_prime,ii) * math.comb(jj + jj_prime, jj) * math.comb(kk + kk_prime, kk) * self.get_area(triangle_vertices[:,0], triangle_vertices[:,1]) / math.comb(2 * degree - 4, self.degree - 2) / math.comb(2 * degree - 2,2)
        self.Mmat = resultmat
        return resultmat
    
    def totalPmat(self):
        resultlist = []
        for kk in range(self.nTriangles):
            Cmat_xx = self.decasteljau_Derivative_Cmat(triangle_vertices=self.tri.points[self.tri.simplices[kk],:],direction='x')
            Cmat_yx = self.decasteljau_Derivative_Cmat(triangle_vertices=self.tri.points[self.tri.simplices[kk],:],direction='yx')
            Cmat_yy = self.decasteljau_Derivative_Cmat(triangle_vertices=self.tri.points[self.tri.simplices[kk],:],direction='y')
            Mmat_tri = self.M_tau(triangle_vertices=self.tri.points[self.tri.simplices[kk],:])
            Pmat_tri = Cmat_xx @ Mmat_tri @ Cmat_xx.T + Cmat_yx @ Mmat_tri @ Cmat_yx.T + Cmat_yy @ Mmat_tri @ Cmat_yy.T
            resultlist.append(Pmat_tri)
        return block_diag(*resultlist)
    
    def fit(self, training_data,  f_obspts, smoothness_constraint = 1, roughness_control = None, sparsity = None, norm_power = 0.5, niter = 100, numofpredictor = None):
        self.numofpredictor = numofpredictor
        
        training_X = training_data[0]
        training_Y = training_data[1]
        self.f_obspts = f_obspts
        
        U,X = np.meshgrid(f_obspts, f_obspts)
        positions = np.vstack([X.ravel(), U.ravel()]).T
        temp = self.data_to_basismat(positions)
        reshape_temp = temp.reshape((f_obspts.shape[0], f_obspts.shape[0], self.totalbasisnum))
        self.reshape_temp = reshape_temp
        #Xdatamat = np.zeros((training_Y.shape[0] * f_obspts.shape[0], self.totalbasisnum)) Single region
        Xdatamat = np.zeros((training_Y.shape[0] * f_obspts.shape[0], self.totalbasisnum, numofpredictor)) #Multiple regions
        
        #Original functions' projection on the the Bbasis          
        for region in range(numofpredictor):
            ind = 0
            for tt in range(training_X.shape[0]):
                for kk in range(f_obspts.shape[0]):
                    tempvec = reshape_temp[kk,:,:] #evaluation of basis functions at a fixed xq
                    projonbasis = np.dot(tempvec.T ,training_X[tt,:,region]).T.ravel()
                    Xdatamat[ind,:,region] = projonbasis
                    ind += 1
                    
                    
        Y_vec = training_Y.ravel()               
        Hmat = self.Hmat(orderR = smoothness_constraint)
        Pmat = self.totalPmat()
        
        gamma_coef_mat = np.zeros(((self.nTriangles,self.nbasispertri, numofpredictor,niter)))
        Y_fittingmat = np.zeros((Y_vec.shape[0], numofpredictor))
        Y_fittingmat[:,0] = Y_vec
        
        for regionid in range(numofpredictor):
            Y_fitting     = np.concatenate((Y_fittingmat[:,regionid], np.zeros(Hmat.shape[0]), np.zeros(Pmat.shape[0])))
            Xdatamat_conc = np.vstack((Xdatamat[:,:,regionid], Hmat, roughness_control * Pmat))
            gamma_initial = np.linalg.inv(Xdatamat_conc.T @ Xdatamat_conc ) @ Xdatamat_conc.T @ Y_fitting
            Y_pred = Xdatamat[:,:,regionid] @ gamma_initial
            gamma_coef_mat[:,:,regionid,0] = gamma_initial.reshape(self.nTriangles,self.nbasispertri)
            if regionid < (numofpredictor - 1):
                Y_fittingmat[:,regionid+1] = Y_fittingmat[:,regionid] - Y_pred
                
        thetas_mat = np.zeros((numofpredictor, self.nTriangles, niter))

        lambda_sparsity = sparsity
        norm_power = 0.5
        tau = (lambda_sparsity * (norm_power**(norm_power)
                                ) * (
                                    (1 - norm_power)**(1 - norm_power)
                                    )
            )**(1/(1 - norm_power))
        indicator_mat     = np.ones((self.nTriangles, self.nbasispertri,numofpredictor,niter))
        indicator_tri_mat =  np.ones((self.nTriangles, numofpredictor, niter))



        for it in range(1,niter):
            #print(it)
            Y_fittingmat = np.zeros((Y_vec.shape[0], numofpredictor))
            Y_fittingmat[:,0] = Y_vec   

            for regid in range(numofpredictor):
                #print(regid)
                
                #Y_fitting     = np.concatenate((Y_fittingmat[:,regid], np.zeros(Hmat.shape[0]), np.zeros(Pmat.shape[0])))
                Y_fitting     = np.concatenate((Y_vec, np.zeros(Hmat.shape[0]), np.zeros(Pmat.shape[0])))
                Xdatamat_conc = np.vstack((Xdatamat[:,:,regid], Hmat, roughness_control * Pmat))
                
                #update theta's with group penalty
                thetas_mat[regid,:,it] = (((1 - norm_power)/(tau * norm_power))**(norm_power)) * np.sum(np.abs(gamma_coef_mat[:, :,regid, it-1])**norm_power,axis = 1) + np.sum(np.abs(gamma_coef_mat[:, :,regid, it-1])**norm_power)
                
                indicator_tri_mat[np.where(thetas_mat[regid,:,it] == 0)[0],regid,it] = 0
                indicator_mat[:,:,regid,it] = (1 - (np.repeat(thetas_mat[regid,:,it], self.nbasispertri) == 0) * 1).reshape(self.nTriangles, self.nbasispertri)
                
                g_s        = (thetas_mat[regid,:,it] ** ( 1- 1/norm_power))
                
                todelete = np.where(indicator_mat[:,:,regid,it].reshape(-1) == 0)[0]
                tokeep   = np.where(indicator_mat[:,:,regid,it].reshape(-1) == 1)[0]
                
                if todelete.shape[0] < self.totalbasisnum:
                    temp_diagvec = 1 /  np.repeat(g_s[thetas_mat[regid,:,it] != 0], self.nbasispertri)
                    G_mat      = (1/Y_vec.shape[0]) * np.diag(temp_diagvec)
                    clf = linear_model.Lasso(alpha=1e-4, fit_intercept=False)

                    sparse_X = Xdatamat_conc[:,tokeep] @ G_mat
                    abc = clf.fit(sparse_X, Y_fitting)
                    
                    temp_coefvec = np.zeros((self.nTriangles * self.nbasispertri))
                    temp_coefvec[tokeep] = abc.coef_ @ G_mat
                    
                    gamma_coef_mat[:,:,regid,it] = temp_coefvec.reshape(self.nTriangles, self.nbasispertri)
                else:
                    temp_coefvec = np.zeros((self.nTriangles * self.nbasispertri))
                    gamma_coef_mat[:,:,regid,it] = temp_coefvec.reshape(self.nTriangles, self.nbasispertri)
                    
                Y_pred = Xdatamat[:,:,regid] @ temp_coefvec
                if regid < (numofpredictor - 1):
                    Y_fittingmat[:,regid+1] = Y_fittingmat[:,regid] - Y_pred
        
        
        sigregion    = np.where(np.sum(indicator_tri_mat[:,:,niter-1],axis=0)/70 == 1)[0]
        if sigregion.shape[0] == 0 :
            return None, None, None, None, None
        else:
            #refit the model with selected regions 
            final_coef  = np.zeros(((self.nTriangles,self.nbasispertri, sigregion.shape[0])))

            Hmat = self.Hmat(orderR = 1) 
            Pmat = self.totalPmat()

            Y_fittingmat = np.zeros((Y_vec.shape[0], sigregion.shape[0]))
            Y_fittingmat[:,0] = Y_vec

            ind = 0
            for regionid in sigregion:
                Y_fitting     = np.concatenate((Y_fittingmat[:,ind], np.zeros(Hmat.shape[0]), np.zeros(Pmat.shape[0])))
                Xdatamat_conc = np.vstack((Xdatamat[:,:,regionid], Hmat, roughness_control * Pmat))
                gamma_initial = np.linalg.inv(Xdatamat_conc.T @ Xdatamat_conc ) @ Xdatamat_conc.T @ Y_fitting
                Y_pred = Xdatamat[:,:,regionid] @ gamma_initial
                final_coef[:,:,ind] = gamma_initial.reshape(self.nTriangles,self.nbasispertri)
                if ind < (sigregion.shape[0] - 1):
                    Y_fittingmat[:,ind+1] = Y_fittingmat[:,ind] - Y_pred
                ind += 1


            Y_predmat = np.zeros((Y_vec.shape[0], sigregion.shape[0]))
            ind = 0
            for kk in sigregion:
                Y_predmat[:,ind] = np.dot(Xdatamat[:,:,kk], final_coef[:,:,ind].reshape(-1))
                ind += 1  
            Y_pred_training = np.sum(Y_predmat,1)
            
            
            sig_coef = final_coef
            sig_tri  = indicator_tri_mat[:,:,niter-1]
            sig_region = sigregion
            
            final_coef_mat = np.zeros(((self.nTriangles,self.nbasispertri, numofpredictor)))
            sigreg_id = 0
            for regid in sig_region:
                final_coef_mat[:,:,regid] = sig_coef[:,:,sigreg_id]
                sigreg_id += 1
            self.final_coef_mat = final_coef_mat
    
            return Y_pred_training, sig_coef, final_coef_mat, sig_tri, sig_region
        
    def predict(self, predX):
        Xdatamat_pred = np.zeros(( predX.shape[1], self.totalbasisnum, predX.shape[2]))
        for region in range(predX.shape[2]):
            ind = 0
            for tt in range(predX.shape[0]):
                for kk in range(predX.shape[1]):
                    tempvec = self.reshape_temp[kk,:,:] #evaluation of basis functions at a fixed xq
                    projonbasis = np.dot(tempvec.T ,predX[tt,:,region]).T.ravel()
                    Xdatamat_pred[ind,:,region] = projonbasis
                    ind += 1
                    
        Y_predmat = np.zeros((Xdatamat_pred.shape[0], predX.shape[2]))
        
        for kk in range(predX.shape[2]):
            Y_predmat[:,kk] = np.dot(Xdatamat_pred[:,:,kk], self.final_coef_mat[:,:,kk].ravel())
        Y_pred = np.sum(Y_predmat,1)
        
        return Y_pred
        
        
                