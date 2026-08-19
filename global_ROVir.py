import numpy as np
from scipy.linalg import orth
from scipy.linalg import eigh
from scipy.linalg import norm
import matplotlib
import matplotlib.pyplot as plt
import kneed
import scipy as sp
from visualization import plot_metrics, compare_retention, display_virtual_coils

from IPython import get_ipython

if 'ipykernel' in str(get_ipython()):
        matplotlib.use('module://matplotlib_inline.backend_inline')

else:
    matplotlib.use('Qt5Agg')         


# colours for print statements
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

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

def elbow_finder(nc, metric, sensitivity, curve_type, curve_direction):
    '''
    Find the elbow of the curve of a certain metric to define
    the top number of virtual cols to retain.

    Parameters
    ----------
        nc -> int: the number of coils 
        metric -> np.ndarray: a 1 x Nc array of values by which the thresholding is based on
        sensitivity -> float: to specify the aggressiveness of elbow recognition for KneeLocator
        curve_type -> str: to specify the type of curve for KneeLocator
        curve_direction -> str: to specify the end behaviour of data as coils increase for KneeLocator

    Return
    ----------
        kneedle.elbow -> int: the coil number at which the elbow was detected 

    '''
    kneedle = kneed.KneeLocator(np.arange(0,nc), metric, S = sensitivity, curve = curve_type, direction = curve_direction)
    print(kneedle.elbow, metric[int(kneedle.elbow)])
    return kneedle.elbow


def top_nv(eigenvec, nc, brain_covar, face_covar, method, threshold, mode, compute_metric_graphs, show_metric_graphs, save_metric_graphs, plot_slices=None, cur_slice=None):
    '''
    Selects the top eigenvectors based on SIR, coil signal energy, coil interference energy, 
    ROI signal retention, or interference signal retention. 

    Parameters
    ----------
        eigenvec -> np.ndarray: an nc x nc array, whose columns are sorted eigenvectors 
        nc -> int: the number of original eigenvectors 
        brain_covar -> np.ndarray: convariance matrix corresponding to brain region
        face_covar -> np.ndarray: covariance matrix corresponding to face region 
        method -> str: the user-selected coil thresholding method 
        threshold -> int: the manual threshold for selecting top coils; if None, 
        then threshold is automatically determined using elbow_finder

    Return
    ----------
        int: the number of top coils to retain

    '''
    brain_signal = np.real(np.diag(eigenvec.conj().T @ brain_covar @ eigenvec)) # calculate signal from brain region
    face_signal = np.real(np.diag(eigenvec.conj().T @ face_covar @ eigenvec)) # calculate signal form face region
    sirs = brain_signal/(face_signal + 1e-12) # calculate brain signal to face signal ratio (signal to interference ratio)

    brain_retain = []
    face_retain = []

    for i in range(1, nc+1):
        eigenvec_retain = orth(eigenvec[:, :i]) # retain i top eigenvectors
        orth_proj = eigenvec_retain @ eigenvec_retain.conj().T # orthogonal projection matrix
        # calculate current percentage signal of the original image retained from the brain and face regions
        cur_brain_retain = (norm((orth_proj @ brain_covar @ orth_proj), ord = 'fro') / norm (brain_covar, ord = 'fro'))*100 
        cur_face_retain = (norm((orth_proj @ face_covar @ orth_proj), ord = 'fro') / norm (face_covar, ord = 'fro'))*100
        brain_retain.append(cur_brain_retain) # append to list of brain signal percentages in order of least top virtual coils retained
        face_retain.append(cur_face_retain) # append to list of face signal percentages in order of least top virtual coils retained

        # print(f'retaining {i} coils retains {cur_brain_retain}% brain and {cur_face_retain}% face)

    if method == "SIR": # if the signal to interference ratio (brain to face) is chosen as metric
        if threshold: # if given a minimum SIR threshold to meet
            i = -1 
            while i+1 <len(sirs) and sirs[i+1] < threshold: 
                i+=1 # retain coils until the current coil meets the minimum SIR
            coil = i + 1 # want to retain the coil that takes us just over min SIR
        else: # if null, use automatic elbow selection to threshold based on SIR curve
            coil = elbow_finder(nc, sirs, 0.1, "convex", "decreasing")

    elif method == "brain_retained": # if the cumulative brain signal retention is chosen as metric
        if threshold: # if given a minimum brain retention percentage to meet
            i = -1
            while i+1 < len(brain_retain) and brain_retain[i+1] < float(threshold):
                i+=1 # retain coils until the cumulative minimum brain signal is met
            coil = i + 1  # want to retain the coil that takes us just over min brain retention
        else: # if null, use automatic elbow selection to threshold based on brain retention curve
            coil = elbow_finder(nc, brain_retain, 5, "concave", "increasing")

    elif method == "face_retained": # if the cumulative face signal retention is chosen as metric
        if threshold: # if given a maximum face retention percentage limit
            i = -1 
            while i+1 < len(face_retain) and face_retain[i+1] < threshold:
                i+=1 # retain coils until the cumulative max face signal is met
            coil = i # don't want to retain the coil that takes us over the limit
        else: # if null, use automatic elbow selection to threshold based on face retention curve
            coil = elbow_finder(nc, face_retain, 0.1, "convex", "increasing")
    else:
        print(RED + "Invalid threholding method. Do you mean 'SIR', 'brain_retained', or 'face_retained'?" + RESET)
        exit()

    xaxis = np.arange(1, nc+1)

    if compute_metric_graphs and mode == 'global':
        plot_metrics(xaxis, coil, nc, sirs, brain_signal, face_signal, brain_retain, face_retain, show_metric_graphs, save_metric_graphs, '- Global ROVir')

    if compute_metric_graphs and mode == 'slice_by_slice' and plot_slices:
        for plot_slice in plot_slices:
            if cur_slice == plot_slice: 
                plot_metrics(xaxis, coil, nc, sirs, brain_signal, face_signal, brain_retain, face_retain, show_metric_graphs, save_metric_graphs, f'(slice {cur_slice})')

    return coil + 1 # plus 1 because slicing is not inclusive, i.e if eigenvec[:,:coil], coil is not included

def form_virtual_coil_data(top_eigenvec, data):

    '''
    Forms the virtual coil data by computing the linear combination of
    the original data with the top eigenvectors. The virtual coil data
    for the jth virtual coil is given by the linear combination of the 
    original data with the jth eigenvector's elements as the coefficients.
    
    Parameters
    ----------
        top_eigenvec -> np.ndarray: an nc x nv array, with columns as the top nv eigenvectors
        data -> np.ndarray: the orignal raw k-space data, shape (x, y, z, ch)


    Returns 
    ----------
        new_data -> np.ndarray: a 4D array of virtual coil data, shape (x, y, z, nv)
    '''

    nv = top_eigenvec.shape[1] # number of top eigenvectors = number of virtual coils
    new_data = np.zeros((data.shape[0], data.shape[1], nv), dtype = data.dtype)

    new_data = np.tensordot(data, top_eigenvec, axes = ([-1], [0]))

    # lin combo of original coil data with lin combo weights as coef
    # (x, y, ch, z) * (ch, nv) = (x, y, z, nv)
    return new_data

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

    A = np.zeros((nc, nc), dtype=np.complex64) # array to store A covariance matrix
    B = np.zeros((nc, nc), dtype=np.complex64) # array to store B covariance matrix

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

def global_rovir(inputs, nc, data, dataID, method, threshold, maskA, maskB, 
                compute_metric_graphs, show_metric_graphs, save_metric_graphs, 
                compute_compare_retention, show_compare_retention, save_compare_retention,
                compute_virtual_coil_images, show_virtual_coil_images, save_virtual_coil_images,
                ref_data=None):

    if "reference_data" in inputs: # compute the image using refernce data
        og_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(ref_data, axes = (0,1,2)), axes=(0,1,2), overwrite_x=True), axes = (0,1,2)) # get image, shape (x, y, z, ch)
    else: # compute the image using target data
        og_image = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(data, axes = (0,1,2)), axes=(0,1,2), overwrite_x=True), axes = (0,1,2)) # get image, shape (x, y, z, ch)

    brain_covar, face_covar = make_A_B(og_image, nc, maskA, maskB)

    eigenvec = rovir(nc, brain_covar, face_covar)

    # plot the retention comparison 
    if compute_compare_retention:
        compare_retention(data, eigenvec, brain_covar, face_covar, nc, data.shape[0]//2, show_compare_retention, save_compare_retention)

    if "top_coils" not in inputs["coil_selection"]: # choose based on heuristics the number of eigenvectors to retain
        top_eigenvec = top_nv(eigenvec, nc, brain_covar, face_covar, method, threshold, 'global', compute_metric_graphs, show_metric_graphs, save_metric_graphs)
    
    else: # if user specifies the number of top virtual coils to keep, use that 
        top_eigenvec = inputs["coil_selection"]["top_coils"]

    print(f'The top nv eigenvectors contain the first {top_eigenvec} eigenvectors')
    eigenvec_retain = eigenvec[:,:top_eigenvec] # retain only the top eigenvectors 
    eigenvec_retain = orth(eigenvec_retain)

    orth_proj = eigenvec_retain @ eigenvec_retain.conj().T # find orthogonal projection matrix for span of retained eigenvectors
   
    # calculate signal retained from maskA region
    brain_retain = (norm((orth_proj @ brain_covar @ orth_proj), ord = 'fro') / norm (brain_covar, ord = 'fro'))*100
    print(GREEN + f'BRAIN SIGNAL RETAINED:{brain_retain}')

    # calculate signal retained from maskB region
    face_retain = (norm((orth_proj @ face_covar @ orth_proj), ord = 'fro') / norm (face_covar, ord = 'fro'))*100
    print(f'FACE SIGNAL RETAINED:{face_retain}' + RESET)

    # forming virtual coils, with eigenvectors as linear combo weights
    virtual_coil_data = form_virtual_coil_data(eigenvec_retain, data) 
    print(GREEN + f'Virtual coils successfully formed' + RESET)

    # compute the image for virtual coils and display
    if compute_virtual_coil_images:
        virtual_coil_img = sp.fft.fftshift(sp.fft.ifftn(sp.fft.fftshift(virtual_coil_data, axes = (0,1,2)), axes=(0,1,2), overwrite_x=True), axes = (0,1,2)) # get image, shape (x, y, z, ch)
        display_virtual_coils(virtual_coil_img, data.shape[0]//2, '', show_virtual_coil_images, save_virtual_coil_images,0)

    # save final defaced raw data
    if inputs["output"]["save_defaced_kspace"] == True:
        np.save(f'results/defaced_{dataID}.npy', virtual_coil_data)

    return virtual_coil_data, top_eigenvec, brain_retain, face_retain