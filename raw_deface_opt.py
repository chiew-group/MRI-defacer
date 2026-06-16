# importing libraries
import json
import os
import nibabel as nib
import numpy as np 
import scipy as sp
from scipy.linalg import orth
from scipy.linalg import norm
from skimage.morphology import convex_hull_image 
from skimage.morphology import binary_erosion 
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import matplotlib
import pandas as pd

from ROVir import rovir
from ROVir import top_nv
from ROVir import form_virtual_coil_data

from visualization import display_defaced
from visualization import compare_retention
from visualization import show_virtual_coils

from to_nifti import niftify

from automask import gen_mask

# colours for print statements
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def rsos(image):
    '''
    Computes the root sum of squares of the image across the channel dimension.
    
    Parameters
    ----------
        image -> np.ndarray: a 4D array of MRI image data, shape (x, y, z, ch)

    Returns
    ----------
        np.ndarray: a 3D array of the root sum of squares of the data, shape (x, y, z)
    '''
    
    return np.sqrt(np.sum(np.abs(image)**2, axis=3))


def set_masks(ROI, interference, option, gap=10):

    '''
    Manipulates the segmentation outputs to simplify shape. 
    
    Parameters
    ----------
        ROI -> np.ndarray: a 3D binary mask representing brain region, shape (x, y, z)
        interference -> np.ndarray: a 3D binary mask representing face region, shape (x, y, z)
        gap -> int: how wide of a gap between ROI and interference is wanted

    Returns
    ----------
        maskA -> np.ndarray: a 3D array of the brain region simplified
        maskB -> np.ndarray: a 3D array of the face region simplified 

    '''
    # ============================== OPTION A ==============================
    if option == "A":
        
        # simplify shape using convex hull
        maskA = convex_hull_image(ROI).astype(int)
        maskB = convex_hull_image(interference).astype(int)

        # get rid of overlap between masks by shrinking face mask
        overlap = maskA.astype(bool) & maskB.astype(bool) 
        while np.count_nonzero(overlap) != 0:
            _, y_face_true, z_face_true = np.nonzero(maskB) # find indices where face mask is nonzero
            zmax_face = np.max(z_face_true) # find top of face mask (closest to brain)
            ymin_face = np.min(y_face_true) # find edge of face mask (closest to neck)
            maskB[:, ymin_face, :] = 0 
            maskB[:, :, zmax_face] = 0
            overlap = maskA.astype(bool) & maskB.astype(bool) 

        # create addition gap by shrinking face mask
        for _ in range(gap):
            _, y_face_true, z_face_true = np.nonzero(maskB)
            zmax_face = np.max(z_face_true)
            ymin_face = np.min(y_face_true)
            maskB[:, ymin_face, :] = 0
            maskB[:, :, zmax_face] = 0

    # ============================== OPTION B ==============================
    if option == "B":
        x_roi, y_roi, z_roi = np.where(ROI)
        
        # get min and max brain index for each direction
        xmin, xmax = np.min(x_roi), np.max(x_roi)
        ymin, ymax = np.min(y_roi), np.max(y_roi)
        zmin, zmax = np.min(z_roi), np.max(z_roi)
        
        # create box mask for brain
        maskA = np.zeros_like(ROI)
        maskA[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = 1

        x_face, y_face, z_face = np.where(interference)
        
        # get min and max face index for each direction
        xmin, xmax = np.min(x_face), np.max(x_face)
        ymin, ymax = np.min(y_face), np.max(y_face)
        zmin, zmax = np.min(z_face), np.max(z_face)
        
        # create box mask for face
        maskB = np.zeros_like(ROI)
        maskB[xmin:xmax+1, ymin:ymax+1, zmin:zmax+1] = 1

        overlap = maskA.astype(bool) & maskB.astype(bool)
        maskB = (maskB.astype(bool) & ~overlap).astype(int)
    
    return maskA, maskB

def make_A_B(image, nc, maskA, maskB):
    '''
    To compute the covariance matrices A and B that correspond to the brain and face regions, respectively.

    Parameters
    ----------
        image -> np.ndarray: the reconstructed image with shape (x, y, z, ch)
        nc -> int: the number of original coils 
        maskA -> np.ndarray: a 3D array of the brain region
        maskB -> np.ndarray: a 3D array of the face region

    Returns
    ----------
        A -> np.ndarray: the 2D covariance matrix corresponding to the brain, shape (nc, nc)
        B -> np.ndarray: the 2D covariance matrix corresponding to the face, shape (nc, nc)

    '''

    A = np.zeros((nc, nc), dtype=np.complex128) # array to store A covariance matrix
    B = np.zeros((nc, nc), dtype=np.complex128) # array to store B covariance matrix

    for z_slice in range(image.shape[2]): # calculate covariance matrices slice by slice
        cur_slice = image[:, :, z_slice, :] # get current z slice
        maskA_slice = maskA[:, :, z_slice] # get current maskA (brain) for z slice
        maskB_slice = maskB[:, :, z_slice] # get current maskB (face) for z slice

        maskedA_slice = cur_slice * maskA_slice[:, :, None] # apply brain mask to z slice of image  
        maskedB_slice = cur_slice * maskB_slice[:, :, None] # apply face mask to z slice of image

        # dot product over x and y axes, then sum over z axis to find covariance
        A += np.tensordot(np.conj(maskedA_slice), maskedA_slice, axes=([0,1],[0,1]))
        B += np.tensordot(np.conj(maskedB_slice), maskedB_slice, axes=([0,1],[0,1]))

    return (A, B)

def by_channel_fft(data): 
    # fft by channels and compute 3d image right away, 
    # getting rid of channel dimension right away
    rsos_image = np.zeros(data.shape[:3])
    for ch in range(data.shape[3]):
        ch_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(data[:, :, :, ch], axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
        rsos_image += np.abs(ch_image)**2
    rsos_image = np.sqrt(rsos_image)
    return rsos_image




def run_raw_deface(config='config.json'):

    matplotlib.use('Qt5Agg')

    # create required folders if not already existing
    os.makedirs('input', exist_ok = True)
    os.makedirs('segmentations', exist_ok = True)
    os.makedirs('results', exist_ok = True)

    # loading inputs from config file
    with open(config, 'r') as config_file:
        inputs = json.load(config_file)

    # loading the numpy raw k-space data and data ID
    data = np.load(inputs["input_data"]["data_path"]) 
    dataID = inputs["input_data"]["data_id"]

    print(GREEN + "Data has been successfully loaded" + RESET)

    # flipping data based on affine
    a11 = float(inputs["input_data"]["a11"])
    a22 = float(inputs["input_data"]["a22"])
    a33 = float(inputs["input_data"]["a33"])

    if a11 < 0: 
        data = data[::-1, :, :, :]

    if a22 < 0:
        data = data[:, ::-1, :, :]

    if a33 < 0:
        data = data[:, :, ::-1, :]

    # moving data axes to (x, y, z, ch) shape
    x_y_z_ch = inputs["input_data"]["x_y_z_channel"]
    sorted_ind = np.argsort(x_y_z_ch)
    data = np.moveaxis(data, [0, 1, 2, 3], sorted_ind) 

    # defining image slices to visualize  
    if "x_slice" in inputs["visualization"]:
        try:
            x_slice = int(inputs["visualization"]["x_slice"])
        except ValueError:
            print(RED + "Your x slice is not a valid integer. Change in config.json." + RESET)
            exit() 
        if x_slice >= data.shape[0] or x_slice < -data.shape[0]:
            print(RED + f"Your x slice is out of bounds for image shape: {data.shape}"  + RESET )
            exit()
    else:
        x_slice = data.shape[0]//2
    
    if "y_slice" in inputs["visualization"]:
        try:
            y_slice = int(inputs["visualization"]["y_slice"])
        except ValueError:
            print(RED + "Your y slice is not a valid integer. Change in config.json." + RESET)
            exit() 
        if y_slice >= data.shape[1] or y_slice < -data.shape[1]:
            print(RED + f"Your y slice is out of bounds for image shape: {data.shape}"  + RESET )
            exit()   
    else:
        y_slice = data.shape[1]//2

    if "z_slice" in inputs["visualization"]:
        try:
            z_slice = int(inputs["visualization"]["z_slice"])
        except ValueError:
            print(RED + "Your z slice is not a valid integer. Change in config.json." + RESET)
            exit() 
        if z_slice >= data.shape[2] or z_slice < -data.shape[2]:
            print(RED + f"Your z slice is out of bounds for image shape: {data.shape}"  + RESET )
            exit()  
    else:
        z_slice = data.shape[2]//2

    # fourier transform and shift data, use scipy with overwriting 
    og_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(data, axes = (0, 1, 2)), axes=(0, 1, 2), overwrite_x=True), axes = (0, 1, 2))
    og_image_rsos = rsos(og_image) # calculate rsos of image

###
    if "reference_data" in inputs: 
        ref_data = np.load(inputs["reference_data"]["data_path"])
        ref_dataID = np.load(inputs["reference_data"]["data_id"])

        print(GREEN + "Reference data has been successfully loaded" + RESET)

        # flipping data based on affine
        a11_ref = float(inputs["reference_data"]["a11"])
        a22_ref = float(inputs["reference_data"]["a22"])
        a33_ref = float(inputs["reference_data"]["a33"])

        if a11_ref < 0: 
            ref_data = ref_data[::-1, :, :, :]

        if a22_ref < 0:
            ref_data = ref_data[:, ::-1, :, :]

        if a33_ref < 0:
            ref_data = ref_data[:, :, ::-1, :]

        # moving data axes to (x, y, z, ch) shape
        x_y_z_ch_ref = inputs["reference_data"]["x_y_z_channel"]
        sorted_ind_ref = np.argsort(x_y_z_ch_ref)
        ref_data = np.moveaxis(ref_data, [0, 1, 2, 3], sorted_ind_ref) 
        
        # fourier transform and shift data 
        ref_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(ref_data, axes = (0, 1, 2)), axes=(0, 1, 2), overwrite_x=True), axes = (0, 1, 2))
        ref_image_rsos = rsos(ref_image) # calculate rsos of image

###

    try:
        gap = int(inputs["masks"]["gap"])
    except ValueError:
        print(RED + "Your gap is not a valid integer. Change in config.json." + RESET)
        exit()

    if "face_mask" not in inputs["masks"] and "brain_mask" not in inputs["masks"]: 

        if "reference_data" in inputs:
            dataID = ref_dataID
            niftify(ref_image_rsos, abs(a11), abs(a22), abs(a33), f'input/input_image_{dataID}') # save nifti of image
            gen_mask(f'input/input_image_{dataID}.nii.gz', f'segmentations/output_mask_{dataID}', dataID) # send nifti image for segmentation

        else:
            niftify(og_image_rsos, abs(a11), abs(a22), abs(a33), f'input/input_image_{dataID}') # save nifti of image
            gen_mask(f'input/input_image_{dataID}.nii.gz', f'segmentations/output_mask_{dataID}', dataID) # send nifti image for segmentation
        
        #face_mask = np.load(f'segmentations/face_mask_{dataID}.npy') # load face mask
        #brain_mask = np.load(f'segmentations/brain_mask_{dataID}.npy') # load brain mask
        face_mask = nib.load(os.path.join(f'segmentations/output_mask_{dataID}', "face.nii.gz")).get_fdata()
        brain_mask = nib.load(os.path.join(f'segmentations/output_mask_{dataID}', "brain.nii.gz")).get_fdata()
    
    else: # using predefined masks
        print(GREEN + "Using predefined masks" + RESET)
        face_mask = nib.load(inputs["masks"]["face_mask"]).get_fdata() # load face mask
        brain_mask = nib.load(inputs["masks"]["brain_mask"]).get_fdata() # load brain mask

    print(GREEN + "Mask has been successfully loaded" + RESET)

    if "manipulation" in inputs["masks"]: # select and apply mask manipulation scheme
        maskA, maskB = set_masks(brain_mask, face_mask, inputs["masks"]["manipulation"], gap) # apply additional masking scheme
        niftify(maskA, abs(a11), abs(a22), abs(a33), f'segmentations/output_mask_{dataID}/brain2.nii.gz')
        niftify(maskB, abs(a11), abs(a22), abs(a33), f'segmentations/output_mask_{dataID}/face2.nii.gz')

    else:
        maskA = brain_mask
        maskB = face_mask

    print(GREEN + "Masks have been prepared" + RESET)

    nc = ref_data.shape[-1] if "reference_data" in inputs else data.shape[-1] # get total number of original coils

    if "reference_data" in inputs:
        brain_covar, face_covar = make_A_B(ref_image, nc, maskA, maskB) # create covariance matrices with reference data
    else:
        brain_covar, face_covar = make_A_B(og_image, nc, maskA, maskB) # create covariance matrices with input data

    print(GREEN + "Brain and face covariance matrices have been computed" + RESET)

    eigenvec = rovir(nc, brain_covar, face_covar) # finding the eigenvectors for (brain_covar)v = λ(face_covar)v
    np.save(f'results/xfm_{dataID}.npy', eigenvec)
    print(GREEN + "Eigenvectors computed" + RESET)
    
    if inputs["visualization"]["plt_on"] == True:
        if inputs["visualization"]["compare_retention"] == True:
            compare_retention(data, eigenvec, brain_covar, face_covar, nc, x_slice) # visualize comparison between different # of coils retained
        if inputs["visualization"]["show_virtual_coils"] == True:
            show_virtual_coils(data, eigenvec, x_slice) # visualize all individual virtual coils

    method = inputs["coil_selection"]["threshold_method"] # load method/metric for selecting top coils
    threshold = inputs["coil_selection"]["threshold_value"] # load limit/requirement for selecting top coils
    
    if "top_coils" not in inputs["coil_selection"]: # choose based on heuristics the number of eigenvectors to retain
        top_eigenvec = top_nv(eigenvec, nc, brain_covar, face_covar, method, threshold, inputs["visualization"]["plt_on"])
    
    else: # if user specifies the number of top virtual coils to keep, use that 
        top_eigenvec = inputs["coil_selection"]["top_coils"]

    print(f'The top nv eigenvectors contain the first {top_eigenvec} eigenvectors')
    eigenvec_retain = eigenvec[:,:top_eigenvec] # retain only the top eigenvectors 
    eigenvec_retain = orth(eigenvec_retain) # orthonormalize retained eigenvectors to noise-whiten

    orth_proj = eigenvec_retain @ eigenvec_retain.conj().T # find orthogonal projection matrix for span of retained eigenvectors
   
    # calculate signal retained from maskA region
    brain_retain = (norm((orth_proj @ brain_covar @ orth_proj), ord = 'fro') / norm (brain_covar, ord = 'fro'))*100
    print(GREEN + f'BRAIN SIGNAL RETAINED:{brain_retain}')

    # calculate signal retained from maskB region
    face_retain = (norm((orth_proj @ face_covar @ orth_proj), ord = 'fro') / norm (face_covar, ord = 'fro'))*100
    print(f'FACE SIGNAL RETAINED:{face_retain}' + RESET)
    
    # forming virtual coils, with eigenvectors as linear combo weights
    virtual_coil_data = form_virtual_coil_data(eigenvec_retain, data) 
    # print(virtual_coil_data.shape)
    print(GREEN + f'Virtual coils successfully formed' + RESET)

    # save final defaced raw data
    if inputs["output"]["save_defaced_kspace"] == True:
        np.save(f'results/defaced_{dataID}.npy', virtual_coil_data)

        
    ####################################################################
    # FOLLOWING CODE IS FOR VISUALIZATION OF FINAL RESULT

    # compute image from virtual coil data
    defaced_image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
    defaced_image = rsos(defaced_image) # find rsos of virtual coil image data

    if inputs["visualization"]["plt_on"]:
        display_defaced(og_image_rsos, defaced_image, x_slice, y_slice, z_slice, maskA, maskB, brain_retain, face_retain) # display images
    
    # save nifti of results for visualization in 3DSlicer
    if inputs["output"]["save_defaced_image"] == True:
        niftify(defaced_image, abs(a11), abs(a22), abs(a33), f'results/defaced_{dataID}_n={top_eigenvec}_brain={brain_retain}_face={face_retain}')

    quality_log(dataID, top_eigenvec, brain_retain, face_retain, maskA, maskB, og_image_rsos, defaced_image)

def quality_log(dataID, top_eigenvec, brain_retain, face_retain, maskA, maskB, og_image_rsos, defaced_image):
    
    total_voxels = og_image_rsos.shape[0] * og_image_rsos.shape[1] * og_image_rsos.shape[2]
    brain_voxels = 100* np.sum(maskA) / total_voxels # compute brain voxels detected as a percentage of whole
    face_voxels = 100* np.sum(maskB) / total_voxels # compute face voxels detected as a percentage of whole
    
    # appened information to a text file to log output metrics
    new_log = pd.DataFrame({
    'Data ID': [dataID],
    'Virtual Coils Retained (#)': [top_eigenvec],
    'Brain Volume Detected (%)': [brain_voxels],
    'Face Volume Detected (%)': [face_voxels],
    'Brain Signal Retained (%)': [brain_retain],
    'Face Signal Retained (%)': [face_retain],
    })
    
    new_log.to_csv('output_log.txt', mode='a', header= not os.path.exists('output_log.txt'), sep='\t')


if __name__ == "__main__": 
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config json file')
    args = parser.parse_args()

    run_raw_deface(args.config)

    