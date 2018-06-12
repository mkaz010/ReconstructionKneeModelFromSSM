# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 19:06:37 2018

@author: Mousa Kazemi
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 09:14:37 2017

@author: Mousa Kazemi

Running PLSR analysis:
    Inputs:
        - sex
        - age
        - height
        - body mass
        - markers
    Outputs:
        - bones geometries (pelvis, femur, tibia, fibula, patella)
        - cartilage layers
        
"""


import numpy as np
import scipy
from pylab import *
from gias2.fieldwork.field import geometric_field
from gias2.learning import PCA
import xml.etree.cElementTree as ET

#from gias2.common import vtktools
from gias2.registration import alignment_fitting
from gias2.fieldwork.field.tools import mesh_fitter as mf
from gias2.registration import alignment_fitting as alf
from gias2.common import transform3D
from gias2.musculoskeletal.bonemodels import bonemodels
from sklearn import cross_decomposition,cross_validation, metrics
from mayavi import mlab
from scipy import stats
import matplotlib.pyplot as plt
import copy
import sys
import datetime

try:
    from gias2.visualisation import fieldvi
    has_mayavi = True
except ImportError:
    has_mayavi = False                            

    
def Extract_LM_Nodal_Coords (Nodal_Coords):

    LM_lst = [87436,86496,87728,56439,53858,41323,39678,82146,63524,68761,67704]

    LM_Nodal_Coords = []
    
    for i in xrange (len(LM_lst)):
        LM_Nodal_Coords.append(Nodal_Coords[LM_lst[i]-1]) 
          
    return LM_Nodal_Coords
    
     
    
pca_output_folder = 'D:\PCA_3\Output'
plsr_data_folder = 'D:\PCA_3/PLSR/data'
plsr_output_folder = 'D:\PCA_3/PLSR/output'    
subject_info = plsr_data_folder+'/subjects_info.txt' 
n = np.loadtxt('D:\PCA_3/PLSR/script/subject_id_file.txt')

#Y = np.load('%s/D_matrix.npy'%pca_output_folder)
Data = np.load('%s/D_NodalCoords_fitted.npy'%pca_output_folder)
Data_flat = []
Data_flat_bn = []
indices = {'Cart_fem':'[: 23800]', 'Cart_tib': '[23800 : 34248]', 'Cart_pat':'[34248 : 38604]', 
           'Bon_fem': '[38604 : 56687]', 'Bon_tib': '[56687 : 84474]', 'Bon_pat': '[84474 :]'}


Ct_fem = Data [:,: 23800, :]
Ct_tib = Data [:, 23800:34248, :]
Ct_pat = Data [:, 34248:38604, :]

Bn_fem = Data [:, 38604:56687, :] # whole bone
Bn_tib = Data [:, 56687:84474, :] # whole bone
Bn_pat = Data      [:, 84474:, :] # whole bone

Bn_fem_partial = Data [:, 48604:56687, :] # trimed for the analysis
Bn_tib_partial = Data [:, 64878:71683, :]  # trimed for the analysis ##### this is the correct trim
Bn_pat_partial = Data      [:, 84474:, :] # trimed for the analysis
#Bone_partial = np.append()


#### --------------- Create responde matrix ---------------###
#respod_mat = Bn_fem
#respod_mat = Bn_tib
#respod_mat = Bn_pat

#respod_mat = np.hstack((Ct_fem, Bn_fem))
#respod_mat = np.hstack((Ct_tib, Bn_tib))
#respod_mat = np.hstack((Ct_pat, Bn_pat))
#respod_mat = Data
#respod_mat = Ct_tib
#respod_mat = Ct_fem
#respod_mat = Ct_pat

meanModel = np.array(respod_mat).mean(axis = 0)

#### -------------- Aligning the segments ---------------- ### 
Segments_aligned = []
inds_w = [(0,23800), (23800,34248), (34248,38604), (38604,56687), (56687,84474), (84474,88004 )]
inds_p = [(0,23800), (23800,34248), (34248,38604), (48604,56687), (64878,71683), (84474,88004 )]
Ts = []        
for segment in [0,1,2,3,4,5]:
    respod_mat= Data[:, inds_p[segment][0]:inds_p[segment][1], :]   
                  
    for i in xrange (0, len(respod_mat)):
        
        source_data = np.array(respod_mat[i])
        target_data = np.array(respod_mat[0])
         
        distal_t0 = target_data.mean(0) - source_data.mean(0)
        
        distal_r0 = np.array([0, 0, 0])
        distal_T0 = np.hstack([distal_t0, distal_r0])
        
        distal_T_opt, aligned_data, reg1_errors  = mf.fitting_tools.fitRigid( 
                source_data, target_data, t0=distal_T0, xtol=1e-5, maxfev=0,
                sample=2000, verbose=0, epsfcn=0, outputErrors=1 )
     
        Ts.append (distal_T_opt)
        
## using cartilage to align models        
Ts[3] =  Ts[0]
Ts[4] =  Ts[1]
Ts[5] =  Ts[2]
       
for segment in [0,1,2,3,4,5]:
    respod_mat_w = Data[:, inds_w[segment][0]:inds_w[segment][1], :]
    
    Source_data_aligned = []   
    for i in xrange (0, len(respod_mat)):        
        data = np.array(respod_mat_w[i])
        dataFitted = transform3D.transformRigid3DAboutCoM(data, Ts[segment])
        Source_data_aligned.append(dataFitted)
    Segments_aligned.append(Source_data_aligned)        

a1 = np.hstack((Segments_aligned[0],Segments_aligned[1]))
a2 = np.hstack ((a1,Segments_aligned[2]) )
a3 = np.hstack ((a2,Segments_aligned[3]) )
a4 = np.hstack ((a3,Segments_aligned[4]) )
a5 = np.hstack ((a4,Segments_aligned[5]) )

Source_data=  a5   

      
for i in xrange (len(Source_data)):
    D_flat = np.ravel(Source_data[i])
    Data_flat.append(D_flat)

for i in xrange (len(Source_data_bn)):
    D_flat_bn = np.ravel(Source_data_bn[i])
    Data_flat_bn.append(D_flat_bn)
        
#for i in xrange (len(Data)):
#    D_flat = np.ravel(Data[i])
#    Data_flat.append(D_flat)

    
meanModel = np.array(Source_data).mean(axis = 0)#.reshape([3,-1,1])
#STDModel = np.array(NodalCoords_fitted).std(axis = 0)
#NodalCoords_fitted_flat = np.array(NodalCoords_fitted_flat).T

NodalCoords_normed = []

for i in xrange (len(Source_data)):
    
    m_normed = np.subtract(Source_data[i], meanModel)
#    NodalCoords_normed.append(np.ravel(m_normed[0:38604]))
    NodalCoords_normed.append(np.ravel(m_normed))
  
NodalCoords_normed = np.array (NodalCoords_normed)

 
# ----------------------These are my predictors------------------------#  
subjects =['01', '02', '04', '06', '08', '09', 10, 12, 13, 15,
       16, 17, 18, 24, 25, 28, 30, 32, 34, 35, 40, 41, 46, ]
       
X = np.load('%s/P_matrix.npy'%plsr_output_folder)[:,:-2] 
params = {'Gender':0, 'Age':1, 'Weight':2, 'Height':3, 'BMI':4,
          'a1':5, 'a3':6, 'a4':7, 'a6':8, 'a7':9, 'a14':10, 'p_angle':11, 'p_ratio':12}
#X = X[:,:5]
#X = X[:,5:]

#### ---------- calculation of bone pcweights ------------ ### 

pca_bn = PCA.PCA()
pca_bn.setData(np.array(Data_flat_bn).T) 
pca_bn.inc_svd_decompose(22) 
pc_bn = pca_bn.PC

modes_bn = [0,1,2,4,5]

X_bn = pc_bn.projectedWeights[0:len(modes_bn),:].T
                              
XX = []                              
for i in xrange (len(X)):
    a = np.append(X[i], X_bn[i])
    XX.append(a)                          
#X = np.array(XX)
                              

pca = PCA.PCA()
pca.setData(np.array(Data_flat).T) # data_matrix shape should be (n_variables, n_samples): OPPOSITE to sklearn
#pca.setData(NodalCoords_normed.T) # data_matrix shape should be (n_variables, n_samples): OPPOSITE to sklearn
pca.inc_svd_decompose(22) # set n_components to something like 5 or 10
pc = pca.PC

meanM = pca.PC.mean.reshape((-1,3))
#	
# This is a function for calculating the RMS error - might be useful	
def reconstructErr( reData, oData ):
	err = ((reData - oData)**2.0).sum(1)
	LSAlign = scipy.sqrt(err)
	return LSAlign

def rmsErr( reData, oData ):
	err = scipy.sqrt((scipy.sqrt(((reData - oData)**2.0).sum(1))**2.0).mean())
	return err
 
def meanErr( reData, oData ):
    err = ((reData - oData)**2.0).sum(1)
    mErr = err.mean()
    sdErr = err.std()   
    return mErr, sdErr, err


modes = [0,1,2,3,4,5,6,7,8,9]

Y = pc.projectedWeights[0:len(modes),:].T


#### Generating final models

Recon_models = []
for train_index, test_index in loo:
    test_index = test_index[0]
    X_train, X_test = X[train_index], X[test_index]
#    Y_train, Y_test = Y[train_index], Y[test_index]
    # initialize plsr
    plsr = cross_decomposition.PLSRegression(n_components=5, scale = False)
    plsr.fit(X, Y)
	
    #prediction
    y = (plsr.predict(X_test)).reshape((len(modes),))

    #Reconstruct test instance
#    P = pc.reconstruct(
#			pc.getWeightsBySD(modes, y),
#			modes
#			)
    P = pc.reconstruct(y, modes)
    #Reshape - this will maybe be different for you - hardcoded for mine here
    R = P.reshape((np.shape(Data_flat)[1]/3,3))
    Recon_models.append (R)
    
    
Temp_Feb_file = 'D:/PCA_3/PLSR/output/FE_models/Template_knee_model'    
tree = ET.parse('{}.feb'.format(Temp_Feb_file))    
root = tree.getroot()

for child in root.findall("Material/material"):
    print child.attrib
    
Nodal_Coords_Temp = []             
for child in root.findall(".//Nodes"):

    for i in child:
        t = i.attrib ['id']+ ','+ i.text
        t1 = [float(word) for word in t.split(',')]
        Nodal_Coords_Temp.append(t1)
Nodal_Coords_Temp = np.array(Nodal_Coords_Temp)             
Added_nodes = Nodal_Coords_Temp [88004:,:]    

   
for j in xrange (len(Recon_models)):
    
#    t = transform3D.directAffine (Nodal_Coords_Temp [: 23800, 1:], np.array(Recon_models)[j][: 23800, :])
    t = transform3D.directAffine (Nodal_Coords_Temp [: 23800, 1:], np.array(Recon_models)[j][: 23800, :])
    Added_nodes_transformed = transform3D.transformAffine(Added_nodes[:,1:], t)
    
    """
    >>>>>>>>> Reconstruct PLS.x_mean_  <<<<<<<<<<<<<<
    """ 
    NodalCoords = np.array(Nodal_Coords_Temp)
    NodalCoords [:88004,1:] = Recon_models[j]
    NodalCoords [88004:,1:] = Added_nodes_transformed
    
    """
    Updating the material properties
    """
    
    LM_Coords_source_updated = Extract_LM_Nodal_Coords (NodalCoords)
    
    for child in root.findall('Material/material'):
        if child.attrib['name'] == 'Material_Femur_Bone_Cart_rigid':
            fem_center_of_mass_old = child[1].text
#            print fem_center_of_mass_old
            
            fem_MEC = LM_Coords_source_updated[3]
            fem_LEC = LM_Coords_source_updated[4]
    
            fem_origin = np.hstack((0, (fem_MEC[1:]+fem_LEC[1:])/2))
            LM_Coords_source_updated = np.vstack ((LM_Coords_source_updated, np.array(fem_origin)))
            
            fem_center_of_mass_new = ','.join(str(x) for x in fem_origin[1:])
            child[1].text = fem_center_of_mass_new
#            print fem_center_of_mass_new
            
            
        elif child.attrib['name'] == 'Material_tibia_Bone_Cart_rigid':
            tib_center_of_mass_old = child[1].text
#            print tib_center_of_mass_old
            
            tibia_LEC = LM_Coords_source_updated[9]
            tibia_MEC = LM_Coords_source_updated[10]
            
            tib_origin = np.hstack ((0, (tibia_MEC[1:]+tibia_LEC[1:])/2))
            LM_Coords_source_updated = np.vstack ((LM_Coords_source_updated, tib_origin))
            
            tib_center_of_mass_new = ','.join(str(x) for x in tib_origin[1:])       
            child[1].text = tib_center_of_mass_new
#            print tib_center_of_mass_new
            
        elif child.attrib['name'] == 'Material_Patella_bone_cart_rigid':
            pat_center_of_mass_old = child[1].text
#            print pat_center_of_mass_old
                            
            patella_inf = LM_Coords_source_updated[0]
            patella_sup = LM_Coords_source_updated[1]
            patella_lat = LM_Coords_source_updated[2]
        
            pat_origin = np.hstack ((0, (patella_inf[1:]+patella_sup[1:])/2))        
            LM_Coords_source_updated = np.vstack ((LM_Coords_source_updated, pat_origin))
            
            pat_center_of_mass_new = ','.join(str(x) for x in pat_origin[1:])
            child[1].text = pat_center_of_mass_new
#            print pat_center_of_mass_new
    
       
            
    i = 0
    for child in root.findall(".//Nodes/node"):
        if i >=len(NodalCoords):
             break
        else:
             old_txt =  child.text 
             new_txt = ','.join(str(d) for d in NodalCoords[i,1:])
             child.text = new_txt
             child.set('updated','yes')
             i+=1                         
          
    tree.write('{}_pls_recon_model_{}.feb'.format(Temp_Feb_file, subjects[j]), encoding="ISO-8859-1", xml_declaration=True)    


    
V = fieldvi.Fieldvi()  

for i in xrange (0, len(Recons)):   
    V.addData(
    	'recon_{}'.format(i+1), Recon_models[i],
     scalar=errFunc, renderArgs={'mode':'point', 'vmin':0, 'vmax':5, 'scale_mode':'none'})
    
    V.addData('model_{}'.format(i+1), Source_data[i], 
    renderArgs={'mode':'sphere', 'scale_factor':2.0, 'color':(0,0,1)})

V.addData('meanModel', meanModel, scalar=errFuncTotal/len(errRMSArray), 
   renderArgs={'mode':'point', 'scale_factor':2.0, 'color':(0,1,1)})   
 
V.configure_traits()
V.scene.background=(1,1,1)









