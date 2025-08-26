import numpy as np
from scipy.linalg import orth
from scipy.linalg import eigh
from scipy.linalg import norm
import matplotlib.pyplot as plt
import kneed

RED = '\033[91m'
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

def top_nv(eigenvec, nc, brain_covar, face_covar, method, threshold):
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

    # visualize metrics as scatterplots
    plt.subplot(2,2,1)
    plt.scatter(xaxis, sirs, c = "blue")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.title("SIR of Each Virtual Coil")
    plt.xlabel("jth Coil")
    plt.legend()

    plt.subplot(2,2,2)
    plt.scatter(xaxis, brain_signal, c = "green")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.title("Signal Energy of Each Virtual Coil")
    plt.xlabel("jth Coil")
    plt.legend()

    plt.subplot(2,2,3)
    plt.scatter(xaxis, face_signal, c = "red")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.title("Interference Energy of Each Virtual Coil")
    plt.xlabel("jth Coil")
    plt.legend()
    
    plt.subplot(2,2,4)
    plt.scatter(np.arange(1, nc+1), np.array(brain_retain), label = "Brain Retained", c= "green")
    plt.scatter(np.arange(1, nc+1), np.array(face_retain), label = "Face Retained", c= "red")
    plt.axvline(x=coil+1, linestyle = "--", label = "Threshold")
    plt.ylabel("Percentage Retained (%)"); plt.xlabel("Total Channels Retained (N)")
    plt.legend()
    plt.title("Brain/Face Signal Retention vs Virtual Coils Retained")
    plt.show()

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
    new_data = np.zeros((data.shape[0], data.shape[1], nv, data.shape[2]), dtype = data.dtype)

    new_data = np.tensordot(data, top_eigenvec, axes = ([3], [0])) 
    # lin combo of original coil data with lin combo weights as coef
    # (x, y, ch, z) * (ch, nv) = (x, y, z, nv)
    return new_data

