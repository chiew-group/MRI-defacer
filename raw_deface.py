import json
import os
import argparse

import nibabel as nib

import numpy as np 
from scipy.linalg import orth
from scipy.linalg import norm

from skimage.morphology import convex_hull_image 

from ROVir import rovir
from ROVir import top_nv
from ROVir import form_virtual_coil_data

from visualization import display_defaced
from visualization import compare_retention
from visualization import show_virtual_coils

from to_nifti import niftify
from automask import gen_mask

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

import matplotlib

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


def set_masks(ROI, interference, gap=10):

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

    return(maskA, maskB)


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
    # to ensure 3D masks now has same dimensions as 4D image data
    # maskA = np.expand_dims(maskA, axis = 3) 
    # maskB = np.expand_dims(maskB, axis = 3)

    # applying masks to image data by entry-wise multiplication
    # maskedA = image*maskA 
    # maskedB = image*maskB 

    maskedA = image*maskA[:,:,:,None]
    maskedB = image*maskB[:,:,:,None]

    # compute the nc x nc covariance matrices
    # A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc))
    # B = np.reshape(maskedB, (-1, nc)).conj().T @ np.reshape(maskedB, (-1, nc))
    A = np.tensordot(np.conj(maskedA), maskedA, axes=([0,1,2],[0,1,2]))
    B = np.tensordot(np.conj(maskedB), maskedB, axes=([0,1,2],[0,1,2]))

    # print(np.linalg.matrix_rank(B)) # checking rank of B

    return (A, B)
    
if __name__ == "__main__": 
    matplotlib.use('Qt5Agg')
    os.makedirs('input', exist_ok = True)
    os.makedirs('segmentations', exist_ok = True)
    os.makedirs('results', exist_ok = True)

    # command-line argument parser for user to specify config file 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to config json file')
    args = parser.parse_args()

    # loading inputs from config file
    with open(args.config, 'r') as config_file:
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
    try:
        x_slice = int(inputs["visualization"]["x_slice"])
        y_slice = int(inputs["visualization"]["y_slice"])
        z_slice = int(inputs["visualization"]["z_slice"])
    except ValueError:
        print(RED + "One of your slices is not a valid integer. Change in config.json." + RESET)
        exit()

    if x_slice >= data.shape[0] or x_slice < -data.shape[0]:
        print(RED + f"Your x slice is out of bounds for image shape: {data.shape}"  + RESET )
        exit()

    if y_slice >= data.shape[1] or y_slice < -data.shape[1]:
        print(RED + f"Your y slice is out of bounds for image shape: {data.shape}"  + RESET)
        exit()

    if z_slice >= data.shape[2] or z_slice < -data.shape[2]:
        print(RED + f"Your z slice is out of bounds for image shape: {data.shape}"  + RESET)
        exit()

    try:
        gap = int(inputs["masks"]["gap"])
    except ValueError:
        print(RED + "Your gap is not a valid integer. Change in config.json." + RESET)
        exit()

    
    # fourier transform and shift data 
    og_image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
    og_image_rsos = rsos(og_image) # calculate rsos of image


    if "face_mask" not in inputs["masks"] and "brain_mask" not in inputs["masks"]: #CHECK WHAT HAPPENS WHEN ONLY ONE IS SPECIFIED, CAN YOU FIND A B WITHOUT ONE 

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

    #maskA, maskB = set_masks(brain_mask, face_mask, gap) # apply additional masking scheme
    #niftify(maskA, abs(a11), abs(a22), abs(a33), f'segmentations/output_mask_{dataID}/brain2.nii.gz')
    #niftify(maskB, abs(a11), abs(a22), abs(a33), f'segmentations/output_mask_{dataID}/face2.nii.gz')
    maskA = brain_mask
    maskB = face_mask
    print(GREEN + "Masks have been prepared" + RESET)

    nc = data.shape[-1] # get total number of original coils
    brain_covar, face_covar = make_A_B(og_image, nc, maskA, maskB) # create covariance matrices
    print(GREEN + "Brain and face covariance matrices have been computed" + RESET)

    eigenvec = rovir(nc, brain_covar, face_covar) # finding the eigenvectors for (brain_covar)v = λ(face_covar)v
    np.save(f'results/xfm_{dataID}.npy', eigenvec)
    print(GREEN + "Eigenvectors computed" + RESET)
    
    # compare_retention(data, eigenvec, brain_covar, face_covar, nc, x_slice) # visualize comparison between different # of coils retained
    #show_virtual_coils(data, eigenvec, x_slice) # visualize all individual virtual coils

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

