import numpy as np
from scipy.linalg import orth
from scipy.linalg import eigh
from scipy.linalg import norm
import matplotlib.pyplot as plt
import kneed
from scipy.signal import savgol_filter


#notes 
    # orthonormalize the eigenvectors in rovir function 
    # define mask such that when totalseg gives you face mask, take entire block as mask
    # get rid of all orthonormalization 

def rovir(nc, A, B):
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
    D, V = eigh(A, B) # compute a vector eigenvalues D and a matrix of eigenvectors V as columns
    # V[:, i] is the eigenvector corresponding to D[i]

    #find unit norm 
    for i in range(nc):
        norm = np.linalg.norm(V[:, i])
        V[: ,i] = V[: ,i] / norm

    # print(V)

    i = np.argsort(D)[::-1] # get indices that would sort eigenvalues in descending order

    V = V[:, i] # rank eigenvectors by sorted eigenvalues
    # take each column of V unit normalize the eigenvectors to 1 

    return V

def elbow_sir(nc, sirs_vec):
    '''
    Find the elbow of the SIR exponential decay curve to set SIR threshold. 

    Parameters
    ----------
        nc -> int: the number of coils 
        sirs_vec -> np.ndarray: a 1 x Nc array of SIR values for each virtual coil

    '''

    # kneedle = kneed.KneeLocator(np.arange(0,nc), sirs_vec, S = 0.5, curve = "convex", direction = "increasing" ) #works for og mask
    kneedle = kneed.KneeLocator(np.arange(0,nc), sirs_vec, S = 0.7, curve = "convex", direction = "increasing" )

    print(kneedle.elbow, sirs_vec[int(kneedle.elbow)])
    return (kneedle.elbow, sirs_vec[int(kneedle.elbow)])

    # kneedle = kneed.KneeLocator(np.arange(0,nc), sirs_vec, S = 10, curve = "concave", direction = "increasing" )
    # print(kneedle.elbow + 1, sirs_vec[int(kneedle.elbow)])
    # return (kneedle.elbow + 1, sirs_vec[int(kneedle.elbow)])


def top_nv_sir(V, nc, A, B, sir_threshold):
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
    # print((A @ V)[:,0])
    # print((A @ V[:,0]))

    signal = np.real(np.diag(V.conj().T @ A @ V))
    # sig1 = V[:, 0].conj().T @ A @ V[:, 0]
    # print(sig1)
    # print(np.diag(signal))

    interference = np.real(np.diag(V.conj().T @ B @ V))
    # print(B)
    print("coil interference:", np.diag(interference))

    # sirs = np.diag(np.abs(signal/(interference + 1e-12))) 
    sirs = signal/(interference + 1e-12)
    print("SIR values are:", sirs)
    coil, sir_threshold = elbow_sir(nc, interference)
    
    # tot_sig = []
    # tot_int = []
    # for i in range(1, 1+nc):
    #     tot_sig.append(np.sum(signal[:i]))
    #     tot_int.append(np.sum(interference[:i]))

    xaxis = np.arange(1, nc+1)

    plt.scatter(xaxis, sirs, c = "blue")
    plt.axvline(x=coil, linestyle = "--", label = "Threshold Elbow")
    plt.title("SIR of Each Virtual Coil")
    plt.xlabel("jth Coil")
    plt.legend()
    plt.show()

    plt.scatter(xaxis, signal, c = "green")
    plt.axvline(x=coil, linestyle = "--", label = "Threshold Elbow")
    plt.title("Signal Energy of Each Virtual Coil")
    plt.xlabel("jth Coil")
    plt.legend()
    plt.show()

    plt.scatter(xaxis, interference, c = "red")
    plt.axvline(x=coil, linestyle = "--", label = "Threshold Elbow")
    plt.title("Interference Energy of Each Virtual Coil")
    plt.xlabel("jth Coil")
    plt.legend()
    plt.show()

    return coil + 1

    # sir_threshold = 1

    # if sirs[0] < sir_threshold: 
    #     print('No coil meets the SIR threshold')
    #     exit()

    # for i, sir in enumerate(sirs):
    #     print(sir)
    #     if sir < sir_threshold:
    #         return i
    # return nc

def top_nv_signal_retained(V, nc, A, B, signal_threshold):

    retain_sig = []

    # print(V)

    for i in range(1, nc+1):
        V_retain = V[:, :i]
        #  V_retain = orth(V[:, :i])
        orth_proj = V_retain @ V_retain.conj().T
        num = orth_proj @ A @ orth_proj
        sig_retain = (norm(num, ord = 'fro') / norm (A, ord = 'fro'))*100
        retain_sig.append(sig_retain)
        print(sig_retain)
    #     if sig_retain >= signal_threshold:
    #         return i 
    # return nc

    # coil, threshold = elbow_sir(nc, retain_sig)

    retain_inter = []
    for i in range(1, nc+1):
        V_retain = V[:, :i]
        #  V_retain = orth(V[:, :i])
        orth_proj = V_retain @ V_retain.conj().T
        num = orth_proj @ B @ orth_proj
        inter_retain = (norm(num, ord = 'fro') / norm (B, ord = 'fro'))*100
        retain_inter.append(inter_retain)
        print(inter_retain)

    plt.scatter(np.arange(1, nc+1), np.array(retain_sig), label = "Signal Retained", c= "green")
    plt.scatter(np.arange(1, nc+1), np.array(retain_inter), label = "Interference Retained", c= "red")
    # plt.axvline(x=coil, color='r', linestyle='--', label='Vertical Line')
    plt.ylabel("Percentage Retained %"); plt.xlabel("Total Channels Retained N")
    plt.legend()
    plt.title("Cumulative Percentage Signal/Interference vs Coils Retained")
    plt.show()

    # return coil+1


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


#for gap, loop while signla of largest coil smaller than 95%. start from gap =0, increase until condition met