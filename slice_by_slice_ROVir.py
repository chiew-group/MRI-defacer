from global_ROVir import form_virtual_coil_data, top_nv
from visualization import display_virtual_coils
import scipy as sp
import numpy as np
from scipy.linalg import eigh, orth, norm


# colours for print statements
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def masking():
    pass

def rovir(nc, brain_covar, face_covar):
    '''
    Finds the eigenvectors from the generalized eigenvalue problem 
    Av = λBv. The eigenvectors are ranked from the largest to the
    smallest eigenvalue, where the largest eigenvalue corresponds
    to the largest signal to interference ratio (sir). This is the 
    ROVir [1] algorithm for localized signal supression.

    Parameters
    ----------
        nc -> int: the number of original eigenvectors/coils
        brain_covar -> np.ndarray: covariance matrix corresponding to brain region 
        face_covar -> np.ndarray: covariance matrix corresponding to face region 

    Returns
    ----------
        eigvec -> np.ndarray: an Nc x Nc array, where columns are eigenvectors
        of the eigenvalue problem Av = λBv sorted by largest eigenvalue to smallest.

    '''
    # solve generalized eigenvalue problem to obtain eigenvalues and eigenvectors 
    # eigval is a vector of eigenvalues, eigvec is a matrix of eigenvectors as columns 
    # eigvec[:, i] is the ith eigenvector corresponding to the eigenvalue eigval[i]
    eigval, eigvec = eigh(brain_covar, face_covar) 

    # find unit norm 
    for i in range(nc):
        norm = np.linalg.norm(eigvec[:, i])
        eigvec[:, i] = eigvec[:, i] / norm

    sorter = np.argsort(eigval)[::-1] # get indices that would sort eigenvalues in descending order
    eigvec = eigvec[:, sorter] # rank eigenvectors by sorted eigenvalues

    return eigvec


def make_A_B(image, maskA, maskB):
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

    # apply masks to image
    maskedA = image*maskA[:, :, None]
    maskedB = image*maskB[:, :, None]
    # print(maskedA.shape)

    # compute covariance matrices A and B
    A = np.tensordot(np.conj(maskedA), maskedA, axes=([0,1],[0,1]))
    B = np.tensordot(np.conj(maskedB), maskedB, axes=([0,1],[0,1]))

    return (A, B)

def is_well_conditioned(matrix, tolerance=1e-6):

    '''
    Checks if matrix is well conditioned.

    Parameters
    ----------
        matrix -> np.ndarray: the matrix being checked
        tolerance -> int: tolerance factor of largest eigenval to be considered significant

    Returns
    ----------
        boolean: True if the smallest eigenvalue is larger than tolerance times largest eigenvalue
    '''
    eigvals = np.linalg.eigvalsh(matrix) # compute eigenvalues
    return eigvals[0] > tolerance * eigvals[-1]   # smallest eigenvalue meaningfully large relative to the biggest

def slice_by_slice_rovir(inputs, nc, data, dataID, method, threshold, readout_axis, maskA, maskB, 
                         compute_metric_graphs, show_metric_graphs, save_metric_graphs, 
                         compute_virtual_coil_images, show_virtual_coil_images, save_virtual_coil_images,
                         ref_data=None):
    
    brain_covars = [] # to store the brain covariance matrix for each slice, list of nc x nc matrices
    face_covars = [] # to store the face covariance matrix for each slice, list of nc x nc matrices
    eigenvecs = [] # to store the eigenvectors for each slice, list of nc x nc matrices
    num_top_eigenvecs = [] # to store the recommended number of top eigenvectors for each slice, list of integers

    # ================= 3. COMPUTE THE ROVIR TRANSFORM SLICE BY SLICE ======================================
    
    if "reference_data" in inputs: # compute hybrid inverse fft using reference data 
        hybrid_fft = sp.fft.fftshift(sp.fft.ifft(sp.fft.fftshift(ref_data, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))
    else: # compute hybrid inverse fft using target data
        hybrid_fft = sp.fft.fftshift(sp.fft.ifft(sp.fft.fftshift(data, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))
    
    # for each readout slice, find the recommended number of top eigenvectors and the eigenvectors
    for slice_num in range(data.shape[readout_axis]): # loop through slices in the readout direction
        
        slice = np.take(hybrid_fft, slice_num, axis=readout_axis)

        full_fft = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(slice, axes = (0,1)), axes=(0,1), overwrite_x=True), axes = (0,1)) # get image, shape (y, z, ch)
            
        A, B = make_A_B(full_fft, np.take(maskA, slice_num, axis=readout_axis), np.take(maskB, slice_num, axis=readout_axis)) # compute covariance matrices from slice image

        brain_covars.append(A) # append brain covariance matrix for current slice
        face_covars.append(B) # append face covariance matrix for current slice
    
        # check the condition of matrix B to ensure no problems are faced in eigh
        if not is_well_conditioned(B) and is_well_conditioned(A) : # if B is rank-deficient and A is not 
            eigenvec = rovir(nc, A, np.eye(nc, dtype=B.dtype)) # solve just Av = λv
        elif not is_well_conditioned(B) and not is_well_conditioned(A): # if B and A are rank-deficient
            eigenvec = np.eye(nc, dtype=A.dtype) # eigenvec for current slice is I so virtual coils for slice are original
        else: eigenvec = rovir(nc, A, B) # finding the eigenvectors for (brain_covar)v = λ(face_covar)v

        eigenvecs.append(eigenvec) # append eigenvec for current slice
    
        if "top_coils" not in inputs["coil_selection"]: # choose based on heuristics the number of eigenvectors to retain
            cur_top_eigenvec = top_nv(eigenvec, nc, A, B, method, threshold, 'slice_by_slice', compute_metric_graphs, show_metric_graphs, save_metric_graphs, f'{dataID}_slice={slice_num}_slice_by_slice_rovir', [data.shape[readout_axis]//2], slice_num)
            num_top_eigenvecs.append(cur_top_eigenvec) # append recommended number of top eigenvecs to retain

    # np.save(f'results/{dataID}_eigenvecs.npy', eigenvecs) 

    if "top_coils" in inputs["coil_selection"]: # if user specifies the number of top virtual coils to keep, use that 
        top_eigenvec = inputs["coil_selection"]["top_coils"]
    else: # if using the automated version, decide on unified number of top eigenvecs to retain
        top_eigenvec = max(num_top_eigenvecs)

    print(GREEN + f'[UPDATE] The top {top_eigenvec} eigenvectors will be retained.' + RESET)

    virtual_coil_data_all_slices = [] # to store the virtual coil data for each slice, a list of (ky, kz, ch) matrices
    virtual_coil_data_all_slices_unaligned = [] # to store the virtual coil data pre-alignment for debugging 

    weighted_mean_brain_retain = 0
    weighted_mean_face_retain = 0

    eigenvecs_aligned = [orth(eigenvecs[0][:, :top_eigenvec]).conj().T]
    
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
        
        eigenvec_retain = orth((eigenvec)[:,:top_eigenvec]) # retain only the top eigenvectors, orthonormalize to noise-whiten
        # eigenvec_retain is a NxM matrix
            
        # =========== for unaligned data =======================================
        cur_virtual_coil_data_unaligned = form_virtual_coil_data((eigenvec)[:,:top_eigenvec], slice)
        # cur_virtual_coil_data_unaligned = form_virtual_coil_data(eigenvec_retain, slice)
        virtual_coil_data_all_slices_unaligned.append(cur_virtual_coil_data_unaligned)
        # ======================================================================

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

    # ======== display virtual coils for unaligned stuff  (delete later) ============

    virtual_hybrid_unaligned = np.stack(virtual_coil_data_all_slices_unaligned, axis=readout_axis) # stack along the x axis to form (x, ky, kz, nv)
    del virtual_coil_data_all_slices_unaligned

    if compute_virtual_coil_images:
        remaining_axes = tuple(ax for ax in (0, 1, 2) if ax != readout_axis) # the two axes still in k-space after the 1D readout ifft
        virtual_coil_data_unaligned = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(virtual_hybrid_unaligned, axes = remaining_axes), axes=remaining_axes, overwrite_x=True), axes = remaining_axes)
        display_virtual_coils(virtual_coil_data_unaligned, 60, 'Virtual Coils Before Alignment', show_virtual_coil_images, save_virtual_coil_images, f'{dataID}_unaligned_xaxis', axis=0) # sagittal view
        display_virtual_coils(virtual_coil_data_unaligned, 60, 'Virtual Coils Before Alignment', show_virtual_coil_images, save_virtual_coil_images, f'{dataID}_unaligned_zaxis', axis=2) # axial view, matches readout_axis

    # ===============================================================
    print(GREEN + f'Mean Brain Signal Retention: {weighted_mean_brain_retain}')
    print(f'Mean Face Signal Retention: {weighted_mean_face_retain}' + RESET)

    virtual_hybrid = np.stack(virtual_coil_data_all_slices, axis=readout_axis) # stack along the x axis to form (x, ky, kz, nv)
    del virtual_coil_data_all_slices

    if compute_virtual_coil_images: # compute images for viewing virtual coils after alignment
        non_readout_axes = tuple(ax for ax in (0, 1, 2) if ax != readout_axis) 
        virtual_coils_img = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(virtual_hybrid, axes = non_readout_axes), axes=non_readout_axes, overwrite_x=True), axes = non_readout_axes)
        display_virtual_coils(virtual_coils_img, 60, 'Virtual Coils After Alignment', show_virtual_coil_images, save_virtual_coil_images, f'{dataID}_xaxis', axis=0) # sagittal view
        display_virtual_coils(virtual_coils_img, 60, 'Virtual Coils After Alignment', show_virtual_coil_images, save_virtual_coil_images, f'{dataID}_zaxis', axis=2) # axial view

    # save virtual hybrid data after phase alignment to check (delete later) ===============
    np.save(f'results/unaligned_{dataID}_readout{readout_axis}.npy', virtual_hybrid_unaligned)
    print(f'Unaligned virtual coil data saved to results/unaligned_{dataID}_readout{readout_axis}.npy')

    np.save(f'results/aligned_{dataID}_readout{readout_axis}.npy', virtual_hybrid)
    print(f'Aligned virtual coil data saved to results/aligned_{dataID}_readout{readout_axis}.npy')
    #========================================================================================

    # fft along the readout axis to obtain fully spatial frequency defaced data
    virtual_coil_data = sp.fft.fftshift(sp.fft.fft(sp.fft.fftshift(virtual_hybrid, axes = (readout_axis)), axis=readout_axis, overwrite_x=True), axes = (readout_axis))
    
    # save final defaced raw data
    if inputs["output"]["save_defaced_kspace"] == True:
        np.save(f'results/defaced_{dataID}.npy', virtual_coil_data)
        print(GREEN + f'[UPDATE] k-space defaced data has been saved to results/defaced_{dataID}.npy' + RESET)

    return virtual_coil_data, top_eigenvec, weighted_mean_brain_retain, weighted_mean_face_retain
