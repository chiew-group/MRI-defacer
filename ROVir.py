import numpy as np
from scipy.linalg import eigh

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

    nc = data.shape[2] #number of channels 

    maskA = np.expand_dims(maskA, axis = 2) # to ensure mask has same dimensions as data
    maskB = np.expand_dims(maskB, axis = 2)

    maskedA = data*maskA
    maskedB = data*maskB

    A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc)) 
    B = np.reshape(maskedB, (-1, nc)).conj().T @ np.reshape(maskedB, (-1, nc))

    # compute a vector eigenvalues D and a matrix of eigenvectors V as columns
    # V[:, i] is the eigenvector corresponding to D[i]
    D,V = eigh(A, B)
    # D,V = np.linalg.eigh(A, B) use this for complex Hermitian symmetric matrices
  
    # get index that would sort eigenvalues in descending order
    i = np.argsort(D)[::-1] 

    # rank eigenvectors by sorted eigenvalues
    V = V[:, i]
    
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
    nc = data.shape[2] #number of channels 

    maskA = np.expand_dims(maskA, axis = 2)
    maskB = np.expand_dims(maskB, axis = 2)

    maskedA = data*maskA
    maskedB = data*maskB

    A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc)) 
    B = np.reshape(maskedB, (-1, nc)).conj().T @ np.reshape(maskedB, (-1, nc)) 

    # signal = np.matrix.H(V)*A*V # Nc x Nc * Nc x Nc * Nc x Nc = Nc x Nc
    # interference = np.matrix.H(V)*B*V #Nc x Nc
    # sir = signal/interference 

    # for i, eigenvec in enumerate(np.transpose(V)):
    #     signal = np.matmul(np.matmul(np.transpose(eigenvec),A),eigenvec)
    #     interference = np.matmul(np.matmul(np.transpose(eigenvec),B),eigenvec)
    #     sir = np.abs(signal/interference)
    #     print(sir)
    #     if sir < sir_threshold:
    #         return i-1
    #     continue
    # return nc-1

    #vectorized version 

    signal = V.conj().T @ A @ V 
    interference = V.conj().T @ B @ V 
    sirs = np.abs(signal/(interference + 1e-12)) # entry-wise division
    sirs = np.diag(sirs) # diagonal of matrix is the vector of sirs
    for i, sir in enumerate(sirs):
        print(sir)
        if sir < sir_threshold:
            return i-1
    return nc - 1

    

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
    new_data = np.zeros((data.shape[0], data.shape[1], nv, data.shape[3]), dtype = data.dtype)

    # for i, eigenvec in enumerate(np.transpose(V)):
    #     # data[x, y, :, z] = [d1, d2, ..., dNc]
    #     # eigenvec = [w1, w2, ..., wNc]
    #     # we want w1*d1 + w2*d2 + ... + wNc*dNc
    #     # print(eigenvec.shape)
    #     new_data[:, :, i, :] = np.tensordot(data, eigenvec, axes = ([2], [0]))
    # return new_data

    # vectorized version
    # with tensordot, the result retains all axes of both input arrays,
    # except the axes that are summed over

    new_data = np.tensordot(data, V, axes = ([2], [0]))
    # (x, y, ch, z) * (ch, Nv) = (x, y, z, Nv)
    new_data = np.moveaxis(new_data, -1, 2)
    return new_data