# importing libraries
import json
import os
import nibabel as nib
import numpy as np 
import scipy as sp
from scipy.linalg import orth
from scipy.linalg import norm
from skimage.morphology import convex_hull_image 
# from skimage.morphology import binary_erosion 
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation

from skimage.draw import line_nd
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

def rsos(image, axis):
    '''
    Computes the root sum of squares of the image across the channel dimension.
    
    Parameters
    ----------
        image -> np.ndarray: a 4D array of MRI image data, shape (x, y, z, ch)

    Returns
    ----------
        np.ndarray: a 3D array of the root sum of squares of the data, shape (x, y, z)
    '''
    
    return np.sqrt(np.sum(np.abs(image)**2, axis=axis))


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

    # ============================== OPTION C ==============================
    # find lowest and most anterior points of brain, draw a tangent 
    if option == "C":
        maskA = convex_hull_image(ROI).astype(int)
        _, y_brain, z_brain = np.where(maskA) # get brain voxel positions along each axes
        pattern = np.zeros(maskA.shape[1:3])
        zmin = np.min(z_brain) # get the most inferior brain voxel position
        ymax = np.max(y_brain) # get the most anterior brain voxel position 

        y_at_zmin = y_brain[z_brain == zmin][0]
        z_at_ymax = z_brain[y_brain == ymax][0]

        # define a tangent going through the two points, fill everything below it 
        slope = (z_at_ymax - zmin)/(ymax - y_at_zmin)

        # edges of image
        y_edge = maskA.shape[1] - 1 # right edge of slice
        z_edge = maskA.shape[2] - 1 # top edge of slice

        y0 = 0 # left of slice
        z_at_y0 = zmin + slope * (y0 - y_at_zmin)

        if z_at_y0 < 0: # z = 0 before y = 0 
            z_at_y0 = 0 
            y0 = (z_at_y0 - zmin) / slope + y_at_zmin
        elif z_at_y0 > z_edge: # z = edge before y = edge
            z_at_y0 = z_edge
            y0 = (z_at_y0 - zmin) / slope + y_at_zmin

        start = (y0, z_at_y0) # left most point of line 

        z_at_y_edge = zmin + slope *(y_edge - y_at_zmin)
        if z_at_y_edge < 0:
            z_at_y_edge = 0
            y_edge = (z_at_y_edge - zmin) / slope + y_at_zmin
        elif z_at_y_edge > z_edge:
            z_at_y_edge = z_edge
            y_edge = (z_at_y_edge - zmin) / slope + y_at_zmin
        
        stop = (y_edge, z_at_y_edge)

        coords = line_nd(start, stop, endpoint=True)
        pattern[tuple(coords)] = 1

        for i in range(len(coords[0])):
            y, z = coords[0][i], coords[1][i]
            pattern[y, :z] = 1

        maskB = np.stack([pattern]* maskA.shape[0], axis = 0) # stack to all x slices
        
        top_layer =[[False, False,  True],
                        [False,  True,  False],
                        [False, False,  False]]

        middle_layer = [[False, False,  True],
                        [False,  True,  False],
                        [False, False,  False]]

        bottom_layer = [[False, False,  True],
                        [False,  True,  False],
                        [False, False,  False]]
        struct_elem = np.array([top_layer, middle_layer, bottom_layer], dtype=bool)

        overlap = maskA.astype(bool) & maskB.astype(bool) 
        while np.count_nonzero(overlap) != 0:
            maskB = binary_erosion(maskB, structure=struct_elem)
            overlap = maskA.astype(bool) & maskB.astype(bool) 

        while gap > 0:
            maskB = binary_erosion(maskB, structure=struct_elem)
            gap -= 1

        maskB = maskB.astype(np.uint8)

    # ============================== OPTION D ==============================
    # L-shaped face mask 
    if option == "D":
        maskA = convex_hull_image(ROI).astype(int)
        _, y_brain, z_brain = np.where(maskA) # get brain voxel positions along each axes

        zmin = np.min(z_brain) # get the most inferior brain voxel position
        ymax = np.max(y_brain) # get the most anterior brain voxel position 

        maskB = np.zeros_like(ROI)

        maskB[:, :, :(zmin-gap)] = 1
        maskB[:, (ymax+gap//2):, :] = 1

    # ============================== OPTION E ==============================
    # complementary masking
    if option == "E": 
        maskA = binary_dilation(ROI, iterations=8).astype(np.uint8)
        maskA = convex_hull_image(maskA).astype(int)
        maskB = np.zeros_like(maskA)
        complement_A = (~binary_dilation(maskA, iterations=gap)).astype(np.uint8)
        y_bound = int(maskB.shape[1]*0.7)
        z_bound = int(maskB.shape[2]*0.7)
        maskB[:, y_bound:, :z_bound] = complement_A[:, y_bound:, :z_bound]

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

    maskedA = image*maskA[:, :, None]
    maskedB = image*maskB[:, :, None]
    print(maskedA.shape)

    # dot product over x and y axes, then sum over z axis to find covariance
    A = np.tensordot(np.conj(maskedA), maskedA, axes=([0,1],[0,1]))
    B = np.tensordot(np.conj(maskedB), maskedB, axes=([0,1],[0,1]))

    return (A, B)

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

    # 1. INVERSE FFT ALONG ALL SPATIAL AXES, RSOS IMAGE, USED TO FIND THE FACE 
    og_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(data, axes = (0, 1, 2)), axes=(0, 1, 2), overwrite_x=True), axes = (0, 1, 2))
    og_image_rsos = rsos(og_image, 3) # calculate rsos of image

    # if reference data exists, load it and process it
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
        ref_image_rsos = rsos(ref_image, 3) # calculate rsos of image

    try:
        gap = int(inputs["masks"]["gap"])
    except ValueError:
        print(RED + "Your gap is not a valid integer. Change in config.json." + RESET)
        exit()

    # 2. GENERATE FACE AND BRAIN MASKS TO DEFINE THE ROVIR TRANSFORM
    # auto generate masks if no predefined masks are given by the user
    if "face_mask" not in inputs["masks"] and "brain_mask" not in inputs["masks"]: 

        # if reference data exists, generate mask based on it
        if "reference_data" in inputs:
            dataID = ref_dataID
            niftify(ref_image_rsos, abs(a11), abs(a22), abs(a33), f'input/input_image_{dataID}') # save nifti of image
            gen_mask(f'input/input_image_{dataID}.nii.gz', f'segmentations/output_mask_{dataID}', dataID) # send nifti image for segmentation

        else: # if no reference data, generate mask based on target data
            niftify(og_image_rsos, abs(a11), abs(a22), abs(a33), f'input/input_image_{dataID}') # save nifti of image
            gen_mask(f'input/input_image_{dataID}.nii.gz', f'segmentations/output_mask_{dataID}', dataID) # send nifti image for segmentation
        
        # load the saved face and brain masks
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

    else: # if no manipulation schemes are specified, use raw masks
        maskA = brain_mask
        maskB = face_mask

    print(GREEN + "Masks have been prepared" + RESET)

    nc = ref_data.shape[-1] if "reference_data" in inputs else data.shape[-1] # get total number of original coils

    brain_covars = [] # to store the brain covariance matrix for each slice, list of nc x nc matrices
    face_covars = [] # to store the face covariance matrix for each slice, list of nc x nc matrices
    eigenvecs = [] # to store the eigenvectors for each slice, list of nc x nc matrices
    num_top_eigenvecs = [] # to store the recommended number of top eigenvectors for each slice, list of integers
    
    method = inputs["coil_selection"]["threshold_method"] # load method/metric for selecting top coils
    threshold = inputs["coil_selection"]["threshold_value"] # load limit/requirement for selecting top coils

    # 3. COMPUTE THE ROVIR TRANSFORM SLICE BY SLICE
    readout_axis = 0 # MAKE A CONFIG INPUT LATER???
    
    if "reference_data" in inputs: # compute hybrid inverse fft using reference data 
        hybrid_fft = sp.fft.fftshift(sp.fft.ifft(sp.fft.fftshift(ref_data, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))
    else: # compute hybrid inverse fft using target data
        hybrid_fft = sp.fft.fftshift(sp.fft.ifft(sp.fft.fftshift(data, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))
    
    # for each readout slice, find the recommended number of top eigenvectors and the eigenvectors
    for slice_num in range(data.shape[readout_axis]): # loop through slices in the readout direction
        
        slice = hybrid_fft[slice_num, :, :, :] # get a single slice in readout direction, shape (ky, kz, ch)
        full_fft = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(slice, axes = (0,1)), axes=(0,1), overwrite_x=True), axes = (0,1)) # get image, shape (y, z, ch)

        A, B = make_A_B(full_fft, nc, maskA[slice_num, :, :], maskB[slice_num, :, :]) # compute covariance matrices from slice image
        brain_covars.append(A) # append brain covariance matrix for current slice
        face_covars.append(B) # append face covariance matrix for current slice

        eigenvec = rovir(nc, A, B) # finding the eigenvectors for (brain_covar)v = λ(face_covar)v
        eigenvecs.append(eigenvec) # append eigenvec for current slice
    
        if "top_coils" not in inputs["coil_selection"]: # choose based on heuristics the number of eigenvectors to retain
            cur_top_eigenvec = top_nv(eigenvec, nc, A, B, method, threshold, inputs["visualization"]["graph"])
            num_top_eigenvecs.append(cur_top_eigenvec) # append recommended number of top eigenvecs to retain

    from collections import Counter

    if "top_coils" in inputs["coil_selection"]: # if user specifies the number of top virtual coils to keep, use that 
            top_eigenvec = inputs["coil_selection"]["top_coils"]
    else: # if using the automated version, decide on unified number of top eigenvecs to retain
        counter = Counter(num_top_eigenvecs)
        top_eigenvec = counter.most_common(1)[0][0] # use the most commen recommendation 
    
    print(GREEN + f'The top {top_eigenvec} eigenvectors will be retained.' + RESET)

    brain_retained = [] # to store the brain retention signal for each slice
    face_retained = [] # to store the face retention signal for each slice
    virtual_coil_data_all_slices = [] # to store the virtual coil data for each slice, a list of (ky, kz, ch) matrices

    # for each readout slice, compute the virtual coil information 
    for slice_num in range(data.shape[readout_axis]):

        eigenvec = eigenvecs[slice_num] # load the eigenvectors for the current slice

        eigenvec_retain = eigenvec[:,:top_eigenvec] # retain only the top eigenvectors 
        eigenvec_retain = orth(eigenvec_retain) # orthonormalize retained eigenvectors to noise-whiten

        orth_proj = eigenvec_retain @ eigenvec_retain.conj().T # find orthogonal projection matrix for span of retained eigenvectors
    
        # calculate signal retained from maskA region
        cur_brain_retain = (norm((orth_proj @ brain_covars[slice_num] @ orth_proj), ord = 'fro') / norm (brain_covars[slice_num], ord = 'fro'))*100
        brain_retained.append(cur_brain_retain)

        # calculate signal retained from maskB region
        cur_face_retain = (norm((orth_proj @ face_covars[slice_num] @ orth_proj), ord = 'fro') / norm (face_covars[slice_num], ord = 'fro'))*100
        face_retained.append(cur_face_retain)
    
        # forming virtual coils, with eigenvectors as linear combo weights
        slice = hybrid_fft[slice_num, :, :, :] # get current slice, shape (ky, kz, ch)
        cur_virtual_coil_data = form_virtual_coil_data(eigenvec_retain, slice) # form virtual coils for this slice
        virtual_coil_data_all_slices.append(cur_virtual_coil_data) 


    mean_brain_retain = np.mean(brain_retained)
    mean_face_retain = np.mean(face_retained)

    print(GREEN + f'Mean Brain Signal Retention: {mean_brain_retain}')
    print(f'Mean Face Signal Retention: {mean_face_retain}' + RESET)

    virtual_hybrid = np.stack(virtual_coil_data_all_slices, axis=0) # stack along the x axis to form (x, ky, kz, nv)
    # fft along the readout axis to obtain fully spatial frequency defaced data
    virtual_coil_data = sp.fft.fftshift(sp.fft.fft(sp.fft.fftshift(virtual_hybrid, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))

    # save final defaced raw data
    if inputs["output"]["save_defaced_kspace"] == True:
        np.save(f'results/defaced_{dataID}.npy', virtual_coil_data)
        
    ####################################################################
    # FOLLOWING CODE IS FOR VISUALIZATION OF FINAL RESULT

    # compute image from virtual coil data
    defaced_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(virtual_coil_data, axes = (0, 1, 2)), axes=(0, 1, 2), overwrite_x=True), axes = (0, 1, 2))
    defaced_image = rsos(defaced_image, 3) # find rsos of virtual coil image data

    if inputs["visualization"]["plt_on"]:
        display_defaced(og_image_rsos, defaced_image, x_slice, y_slice, z_slice, maskA, maskB, mean_brain_retain, mean_face_retain) # display images
    
    # save nifti of results for visualization in 3DSlicer
    if inputs["output"]["save_defaced_image"] == True:
        niftify(defaced_image, abs(a11), abs(a22), abs(a33), f'results/defaced_{dataID}_n={top_eigenvec}_brain={mean_brain_retain:.2}_face={mean_face_retain:.2}')

    mask_scheme = inputs["masks"]["manipulation"] if "manipulation" in inputs["masks"] else None
    quality_log(dataID, top_eigenvec, mean_brain_retain, mean_face_retain, maskA, mask_scheme, method, threshold)

    return nc, data, dataID, og_image, og_image_rsos


def quality_log(dataID, top_eigenvec, brain_retain, face_retain, maskA, mask_scheme, method, threshold):
    
    # total_voxels = og_image_rsos.shape[0] * og_image_rsos.shape[1] * og_image_rsos.shape[2]
    brain_voxels = np.sum(maskA) # compute brain voxels detected as a percentage of whole image
    
    # perform brain segmentation on defaced image to find resulting brain volume 
    gen_mask(f'results/defaced_{dataID}_n={top_eigenvec}_brain={brain_retain:.2}_face={face_retain:.2}.nii.gz', f'segmentations/defaced_mask_{dataID}', dataID)

    defaced_brain_mask = nib.load(f'segmentations/defaced_mask_{dataID}/brain.nii.gz').get_fdata()
    defaced_brain_voxels = np.sum(defaced_brain_mask)

    # appened information to a text file to log output metrics
    new_log = pd.DataFrame({
    'Data ID': [dataID],
    'Masking Method': [mask_scheme],
    'Coil Thresholding Method': [method],
    'Thresholding Value': [threshold],
    'Virtual Coils Retained (#)': [top_eigenvec],
    'Brain Voxels Before Defacing (#)': [brain_voxels],
    'Brain Voxels After Defacing (#)': [defaced_brain_voxels],
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