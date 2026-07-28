import scipy as sp
import json
import numpy as np
from raw_deface_opt import run_raw_deface, set_masks, make_A_B, compute_image, is_well_conditioned
from to_nifti import niftify
from automask import gen_mask
import nibabel as nib
from scipy.linalg import orth
from scipy.linalg import norm
from ROVir import rovir
from ROVir import form_virtual_coil_data

GREEN = '\033[92m'
RESET = '\033[0m'

def backend(input):
    # loading inputs from config file
    with open(input, 'r') as config_file:
        inputs = json.load(config_file)

    # loading the numpy raw k-space data and data ID
    data = np.load(inputs["input_data"]["data_path"]) 
    dataID = inputs["input_data"]["data_id"]

    print(GREEN + "[UPDATE] Data has been successfully loaded" + RESET)

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

    og_image_rsos = compute_image(data) # calculate rsos of image
    nc = data.shape[-1]
    readout_axis = int(inputs["input_data"]["readout_axis"])

    niftify(og_image_rsos, abs(a11), abs(a22), abs(a33), f'input/input_image_{dataID}') # save nifti of image
    gen_mask(f'input/input_image_{dataID}.nii.gz', f'segmentations/output_mask_{dataID}', dataID) # send nifti image for segmentation

    return nc, data, dataID, readout_axis, og_image_rsos

def backend_recompute(nc, data, dataID, readout_axis, mask_option, gap, n_coils):
    # load default face and brain mask
    maskA = (nib.load(f'segmentations/output_mask_{dataID}/brain.nii.gz')).get_fdata()
    maskB = (nib.load(f'segmentations/output_mask_{dataID}/face.nii.gz')).get_fdata()

    if mask_option != 'Default':
        maskA, maskB = set_masks(maskA, maskB, mask_option, gap) # compute manipulated masks

    hybrid_fft = sp.fft.fftshift(sp.fft.ifft(sp.fft.fftshift(data, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))

    brain_covars = [] # to store the brain covariance matrix for each slice, list of nc x nc matrices
    face_covars = [] # to store the face covariance matrix for each slice, list of nc x nc matrices
    eigenvecs = [] # to store the eigenvectors for each slice, list of nc x nc matrices
    
    # for each readout slice, find the recommended number of top eigenvectors and the eigenvectors
    for slice_num in range(data.shape[readout_axis]): # loop through slices in the readout direction
        
        slice = np.take(hybrid_fft, slice_num, axis=readout_axis)

        full_fft = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(slice, axes = (0,1)), axes=(0,1), overwrite_x=True), axes = (0,1)) # get image, shape (y, z, ch)

        maskA = maskA.astype(np.float32)
        maskB = maskB.astype(np.float32)

        A, B = make_A_B(full_fft, nc, np.take(maskA, slice_num, axis=readout_axis), np.take(maskB, slice_num, axis=readout_axis)) # compute covariance matrices from slice image

        brain_covars.append(A) # append brain covariance matrix for current slice
        face_covars.append(B) # append face covariance matrix for current slice
    
        # check the condition of matrix B to ensure no problems are faced in eigh
        if not is_well_conditioned(B) and is_well_conditioned(A) : # if B is rank-deficient and A is not 
            eigenvec = rovir(nc, A, np.eye(nc, dtype=B.dtype)) # solve just Av = λv
        elif not is_well_conditioned(B) and not is_well_conditioned(A): # if B and A are rank-deficient
            eigenvec = np.eye(nc, dtype=A.dtype) # eigenvec for current slice is I so virtual coils for slice are original
        else: eigenvec = rovir(nc, A, B) # finding the eigenvectors for (brain_covar)v = λ(face_covar)v

        eigenvecs.append(eigenvec) # append eigenvec for current slice

    virtual_coil_data_all_slices = [] # to store the virtual coil data for each slice, a list of (ky, kz, ch) matrices

    weighted_mean_brain_retain = 0
    weighted_mean_face_retain = 0

    eigenvecs_aligned = [orth(eigenvecs[0][:, :n_coils]).conj().T]
    # for each readout slice, compute the virtual coil information 
    for slice_num in range(data.shape[readout_axis]):
        
        # get current slice hybrid fft
        if readout_axis == 0:
            slice = hybrid_fft[slice_num, :, :, :] # get current slice, shape (ky, kz, ch)
        elif readout_axis == 1: 
            slice = hybrid_fft[:, slice_num, :, :] # get current slice, shape (ky, kz, ch)
        elif readout_axis == 2:
            slice = hybrid_fft[:, :, slice_num, :] # get current slice, shape (ky, kz, ch)

        eigenvec = eigenvecs[slice_num] # load the eigenvectors for the current slice
        
        eigenvec_retain = orth((eigenvec)[:,:n_coils]) # retain only the top eigenvectors, orthonormalize to noise-whiten
        # eigenvec_retain is a NxM matrix
            
        # perform phase alignment, assuming first slice is aligned
        if slice_num != 0: 
            # nv x nc * nc x nv = nv x nv
            c =  eigenvec_retain.conj().T @ (eigenvecs_aligned[slice_num-1]).conj().T
            U, _, Vh = np.linalg.svd(c) # each is nv x nv
            P = Vh.conj().T @ U.conj().T
            eigenvecs_aligned.append(P @ eigenvec_retain.conj().T) 

        eigenvec_retain = eigenvecs_aligned[slice_num].conj().T
        orth_proj = eigenvec_retain @ eigenvec_retain.conj().T # find orthogonal projection matrix for span of retained eigenvectors
    
        # calculate signal retained from maskA region
        if norm(brain_covars[slice_num], ord = 'fro') != 0:
            cur_brain_retain = (norm((orth_proj @ brain_covars[slice_num] @ orth_proj), ord = 'fro') / norm(brain_covars[slice_num], ord = 'fro'))*100
        else: cur_brain_retain = 0

        # calculate signal retained from maskB region
        if norm(face_covars[slice_num], ord = 'fro') != 0:
            cur_face_retain = (norm((orth_proj @ face_covars[slice_num] @ orth_proj), ord = 'fro') / norm(face_covars[slice_num], ord = 'fro'))*100
        else: cur_face_retain = 0

        weight_brain_cur = np.sum(np.take(maskA, slice_num, axis=readout_axis)) / np.sum(maskA) 
        weight_face_cur = np.sum(np.take(maskB, slice_num, axis=readout_axis)) / np.sum(maskB)

        weighted_mean_brain_retain += weight_brain_cur * cur_brain_retain
        weighted_mean_face_retain += weight_face_cur * cur_face_retain

        # forming virtual coils, with eigenvectors as linear combo weights
        cur_virtual_coil_data = form_virtual_coil_data(eigenvec_retain, slice) # form virtual coils for this slice
        virtual_coil_data_all_slices.append(cur_virtual_coil_data)

    virtual_hybrid = np.stack(virtual_coil_data_all_slices, axis=readout_axis) # stack along the x axis to form (x, ky, kz, nv)
    non_readout_axes = tuple(ax for ax in (0, 1, 2) if ax != readout_axis) 
    defaced_img = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(virtual_hybrid, axes=non_readout_axes), axes=non_readout_axes),axes=non_readout_axes)
    image_rsos = np.sqrt(np.sum(np.abs(defaced_img)**2, axis=3))

    return maskA, maskB, image_rsos, weighted_mean_brain_retain, weighted_mean_face_retain 