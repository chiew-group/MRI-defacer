import numpy as np
from scipy.linalg import eigh
from scipy.linalg import norm
from numpy.linalg import matrix_rank


def rovir(data, maskA, maskB):
    '''
    Finds the eigenvectors from the generalized eigenvalue problem 
    Av = DBv. The eigenvectors are ranked from the largest to the
    smallest eigenvalue, where the largest eigenvalue corresponds
    to the largest SIR. This is the ROVir algorithm for increasing
    signal energy from the ROI and decreasing intereference from
    the uninteresting region.

    Parameters
    ----------
        data -> np.ndarray: a 4D array of MRI data, shape (x, y, ch, z)
        maskA -> np.ndarray: an array for signal region
        maskB -> np.ndarray: an array for interference region

    Returns
    ----------
        V -> np.ndarray: an Nc x Nc array, with right eigenvectors of the 
                        eigenvalue problem Av = DBv 

    '''
    # check if eigenvectors are the same use matlab 
    nc = data.shape[3] #number of channels 

    maskA = np.expand_dims(maskA, axis = 3) # to ensure mask has same dimensions as data
    maskB = np.expand_dims(maskB, axis = 3)

    maskedA = data*maskA # applying masks by entry-wise mult
    maskedB = data*maskB 

    A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc)) # Nc x Nc
    
    B = np.reshape(maskedB, (-1, nc)).conj().T @ np.reshape(maskedB, (-1, nc))
    D, V = eigh(A, B) # compute a vector eigenvalues D and a matrix of eigenvectors V as columns
    # V[:, i] is the eigenvector corresponding to D[i]

    i = np.argsort(D)[::-1] # get indices that would sort eigenvalues in descending order

    V = V[:, i] # rank eigenvectors by sorted eigenvalues

    return V

def top_nv_sir(V, data, maskA, maskB, sir_threshold):
    '''
    Find the number of top Nv < Nc eigenvectors for the linear combination weight. 
    Calculated by first find the signal interference ratio for each coil, 
    then only counting those with a SIR > 1, indicating the signal coming 
    from that coil is greater than the interference. Specfically, find
    signal = w'Aw, interference = w'Bw energy, sir = signal/interference.

    Parameters
    ----------
        V -> np.ndarray: an Nc x Nc array, with right eigenvectors of Av = DBv 
        data -> np.ndarray: a 4D array of MRI data, shape (x, y, ch, z)
        maskA -> np.ndarray: an m x n array for signal region
        maskB -> np.ndarray: an m x n array for interference region
        sir_threshold -> int: threshold sir for top Nv coils
    
    Returns
    ----------
        Nv -> int: the index of the last top Nv coil
    
    '''
    nc = data.shape[3] #number of channels 

    maskA = np.expand_dims(maskA, axis = 3)
    maskB = np.expand_dims(maskB, axis = 3)

    maskedA = data*maskA
    maskedB = data*maskB

    A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc)) 
    B = np.reshape(maskedB, (-1, nc)).conj().T @ np.reshape(maskedB, (-1, nc)) 

    #vectorized version 
    signal = V.conj().T @ A @ V 
    interference = V.conj().T @ B @ V 
    sirs = np.diag(np.abs(signal/(interference + 1e-12))) 
    
    if sirs[0] < sir_threshold: 
        print('No coil meets sir_threshold')
        exit()

    for i, sir in enumerate(sirs):
        print(sir)
        if sir < sir_threshold:
            return i
    return nc

def top_nv_signal_retained(V, data, maskA, maskB, signal_threshold):
    nc = data.shape[3] #number of channels 

    maskA = np.expand_dims(maskA, axis = 3)
    maskB = np.expand_dims(maskB, axis = 3)

    maskedA = data*maskA

    A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc)) 

    for i in range(1, nc+1):
        V_retain = V[:, :i]
        orth_proj = V_retain @ V_retain.conj().T
        num = orth_proj @ A @ orth_proj
        sig_retain = (norm(num, ord = 'fro') / norm (A, ord = 'fro'))*100
        print(sig_retain)
        if sig_retain >= signal_threshold:
            return i 
    return nc

def form_virtual_coil_data(V, data):

    '''
    Forms the virtual coil data by computing the linear combination of
    the original data with the top eigenvectors. The virtual coil data
    for the jth virtual coil is given by the linear combination of the 
    original data with the jth eigenvector's elements as the coefficients.
    
    Parameters
    ----------
        V -> np.ndarray: an Nc x Nv array, with columns as the top Nv eigenvectors
        data -> np.ndarray: a 4D array of MRI data, shape (x, y, ch, z)


    Returns 
    ----------
        new_data -> np.ndarray: a 4D array of virtual coil data, shape (x, y, Nv, z)
    '''

    nv = V.shape[1] # number of top eigenvectors = number of virtual coils
    new_data = np.zeros((data.shape[0], data.shape[1], nv, data.shape[2]), dtype = data.dtype)

    new_data = np.tensordot(data, V, axes = ([3], [0])) 
    # lin combo of original coil data with lin combo weights as coef
    # (x, y, ch, z) * (ch, Nv) = (x, y, z, Nv)
    return new_data