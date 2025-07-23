import json

import numpy as np 
from scipy.ndimage import binary_dilation
from scipy.ndimage import binary_erosion
from scipy.linalg import orth

from ROVir import rovir
from ROVir import top_nv_sir
from ROVir import top_nv_signal_retained
from ROVir import form_virtual_coil_data

from visualization import display

from to_nifti import niftify
from automask import gen_mask

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

def rsos(data):
    '''
    Computes the root sum of squares of the data across the channel dimension.
    
    Parameters
    ----------
        data -> np.ndarray: a 4D array of MRI data, shape (x, y, ch, z)

    Returns
    ----------
        np.ndarray: a 3D array of the root sum of squares of the data, shape (x, y, z)
    '''
    
    return np.sqrt(np.sum(np.abs(data)**2, axis=3))

def expand_mask(brain_mask, maskB):

    #maskB = binary_erosion(maskB, iterations=7) #just in case random outlier points were generated
    x_face, y_face, z_face = np.nonzero(maskB)
    xmaxf, xminf = np.max(x_face), np.min(x_face)
    ymaxf, yminf = np.max(y_face), np.min(y_face)
    zmaxf, zminf = np.max(z_face), np.min(z_face)

    x_brain, y_brain, z_brain = np.nonzero(brain_mask)
    xmaxb, xminb = np.max(x_brain), np.min(x_brain)
    ymaxb, yminb = np.max(y_brain), np.min(y_brain)
    zmaxb, zminb = np.max(z_brain), np.min(z_brain)

    new_maskB = np.zeros((maskB.shape))
 
    # #METHOD 1: STRIPES WITH BRAIN BORDER   

    # new_maskB[:, ymaxb:, :] = 1
    # new_maskB[:, :, :zminb] = 1

    #METHOD 2: BRAIN FACE AVERAGE BORDER STRIPES

    yavg = (ymaxb+yminf) //2
    zavg = (zmaxf+zminb) //2

    # new_maskB[:, yavg:, :] = 1
    # new_maskB[:, :, :zavg] = 1

    #METHOD 3: FACE 4 DIRECTION BOARDER 
    #taking face boundary in each direction to define box
    # new_maskB[xminf:xmaxf, yminf:ymaxf, zminf:zmaxf] = 1

    # #METHOD 4: OVAL MASK USING FACE MASK DIAMETER
    x, y, z = np.meshgrid(np.arange(maskB.shape[0]), np.arange(maskB.shape[1]), np.arange(maskB.shape[2]), indexing = 'ij')
    xcent, ycent, zcent = (xmaxf + xminf) //2, (ymaxf + yminf) //2, (zmaxf+zminf) //2
    xrad, yrad, zrad = (xmaxf - xminf) //2, (ymaxf - yminf) //2, (zmaxf - zminf) //2
    circ = (((x-xcent)/xrad)**2 + ((y-ycent)/yrad)**2 + ((z-zcent)/zrad)**2) <= 1
    overlap = circ.astype(bool) & brain_mask.astype(bool)
    # if np.count_nonzero(overlap) != 0:
    #     x_overlap, y_overlap, z_overlap = np.nonzero(overlap.astype(int))
    #     ymaxo, zmino = np.max(y_overlap), np.min(z_overlap)
    #     circ = (((x-xcent)/xrad)**2 + ((y-ycent)/(ymaxo-ycent))**2 + ((z-zcent)/(zmino-zcent))**2) <= 1
    while np.count_nonzero(overlap) != 0:
        circ = binary_erosion(circ)
        overlap = circ.astype(bool) & brain_mask.astype(bool)

    new_maskB[circ] =1
    
    # # #METHOD 5: start with og mask, binary dilate until it hits the brain 
    # new_maskB = maskB
    # overlap = new_maskB.astype(bool) & brain_mask.astype(bool)
    # i = 0
    # while np.count_nonzero(overlap) ==0:
    #     new_maskB = binary_dilation(new_maskB) 
    #     overlap = new_maskB.astype(bool) & brain_mask.astype(bool)
    #     i+=1
    # print(i)

    return new_maskB

    # x, y, z = np.meshgrid(np.arange(maskB.shape[0]), np.arange(maskB.shape[1]), np.arange(maskB.shape[2]), indexing = 'ij')
    # rect_prism = ((x>= 0) & (x<maskB.shape[0]) & (y >= ymaxb) & (y <maskB.shape[1]) & (z>=0) & (z<zminb))
    # new_maskB[rect_prism] = 1

    # for x in range(maskB.shape[0]):
    #     y, z = np.nonzero(maskB[x, :, :])

    #     if len(y) != 0 and len(z) != 0:
    #         yminf, ymaxf = np.min(y), np.max(y)
    #         zminf, zmaxf = np.min(z), np.max(z)
    #         new_maskB[x, yminf:ymaxf, zminf:zmaxf] = 1

    # for y in range(maskB.shape[1]):
    #     x, z = np.nonzero(maskB[:, y, :])

    #     if len(x) != 0 and len(z) != 0:
    #         xminf, xmaxf = np.min(x), np.max(x)
    #         zminf, zmaxf = np.min(z), np.max(z)
    #         new_maskB[xminf:xmaxf,y, zminf:zmaxf] = 1
    
def make_A_B(image, nc, maskA, maskB):
    maskA = np.expand_dims(maskA, axis = 3) # to ensure mask has same dimensions as data
    maskB = np.expand_dims(maskB, axis = 3)

    maskedA = image*maskA # applying masks by entry-wise mult
    maskedB = image*maskB 

    A = np.reshape(maskedA, (-1, nc)).conj().T @ np.reshape(maskedA, (-1, nc)) # Nc x Nc
    B = np.reshape(maskedB, (-1, nc)).conj().T @ np.reshape(maskedB, (-1, nc))

    return (A, B)
    
if __name__ == "__main__": 

    # loading inputs from config file
    with open('config.json', 'r') as config_file:
        inputs = json.load(config_file)

    # need validation for file type and path 
    data = np.load(inputs["data_path"]) 
    print(GREEN + "Data has been successfully loaded" + RESET)

    # moving data axes to (x, y, z, ch) shape
    x_y_z_ch = inputs["x_y_z_channel"]
    sorted_ind = np.argsort(x_y_z_ch)
    data = np.moveaxis(data, [0, 1, 2, 3], sorted_ind) 

    a11 = float(inputs["a11"])
    a22 = float(inputs["a22"])
    a33 = float(inputs["a33"])

    if a11 < 0: 
        data = data[::-1, :, :, :]

    if a22 < 0:
        data = data[:, ::-1, :, :]

    if a33 < 0:
        data = data[:, :, ::-1, :]

    # defining image slices
    try:
        x_slice = int(inputs["x_slice"])
        y_slice = int(inputs["y_slice"])
        z_slice = int(inputs["z_slice"])
    except ValueError:
        print(RED + "One of your slices is not a valid integer. Change in config.json." + RESET)
        exit()

    if x_slice >= data.shape[0] or x_slice < -data.shape[0]:
        print(RED + "Your x slice is out of bounds. Check your image shape."  + RESET )
        exit()

    if y_slice >= data.shape[1] or y_slice < -data.shape[1]:
        print(RED + "Your y slice is out of bounds. Check your image shape."  + RESET)
        exit()

    if z_slice >= data.shape[2] or z_slice < -data.shape[2]:
        print(RED + "Your z slice is out of bounds. Check your image shape."  + RESET)
        exit()

    try:
        gap = int(inputs["gap"])
    except ValueError:
        print(RED + "Your gap is not a valid integer. Change in config.json." + RESET)
        exit()

    # fourier transform and shift data 
    # image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
    image = np.fft.ifftn(data, axes = (0,1,2)) #ATTENTION: ONLY FOR THE CALGARY DATASET, WHICH WAS ALREADY SHIFTED
    image_rss = rsos(image)

    # niftify(image_rss, abs(a11), abs(a22), abs(a33))
    # gen_mask(r"results\input_image.nii", r"results\output_mask.nii") 
    
    maskBface = np.load(r"results\face_mask.npy") # define face mask
    maskBfat = np.load(r"results\fat_mask.npy") # define face mask
    maskBmusc = np.load(r"results\muscle_mask.npy") # define face mask
    brain_mask = np.load(r"results\brain_mask.npy")

    print(GREEN + "Mask has been successfully loaded" + RESET)
    maskB = maskBface #+ maskBmusc#maskBfat + maskBmusc
    maskB = expand_mask(brain_mask, maskBface)
    maskA = np.ones((data.shape[0], data.shape[1], data.shape[2])) # define signal mask
    # print(np.count_nonzero(maskBface), np.count_nonzero(maskBfat), np.count_nonzero(maskBmusc))

    # maskA = brain_mask
    maskA = maskA - maskB

    # maskA = brain_mask 
    # maskB = np.ones(data.shape[:3]) - maskA

    # if gap >= 1: # binary_erosion iterates all is gone if iteration is less than 1
    #     maskA = maskA - binary_dilation(maskB, iterations=gap)
    #     maskB = binary_erosion(maskB, iterations=gap)
    # else:
    #     maskA = maskA - maskB

    # # manual mask
    # maskA = np.ones(data.shape[:3])
    # maskB = np.ones(data.shape[:3])
    # maskA[:data.shape[0]//2, :data.shape[1], :data.shape[2]] = 0
    # maskB = maskB - maskA

    # maskA = np.ones(data.shape[:3])
    # maskB = np.zeros(data.shape[:3])
    # x_start, x_end = 0, data.shape[0] 
    # y_start, y_end = 110, data.shape[1]
    # z_start, z_end = 0, 85
    # gap = 0
    # maskA[x_start:x_end, y_start:y_end, z_start:z_end] = 0
    # maskB[x_start-gap:x_end+gap, y_start-gap:y_end+gap, z_start-gap:z_end+gap]= 1

    nc = data.shape[-1]
    A, B = make_A_B(image, nc, maskA, maskB)

    V = rovir(nc, A, B) # finding the eigenvectors for Av = λBv
    print(GREEN + "Eigenvectors computed" + RESET)

    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    # # displaying all virtual coils
    # all_coils = form_virtual_coil_data(V, data)
    # # image_all = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(all_coils, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
    # image_all = np.fft.ifftn(all_coils, axes = (0, 1, 2)) #ATTENTION: ONLY FOR THE CALGARY DATASET, WHICH WAS ALREADY SHIFTED
    # figure = plt.figure(figsize = (10, 10))
    # for coil in range(all_coils.shape[3]):
    #     plt.subplot(4, 3, coil+1)
    #     plt.imshow(np.abs(image_all[x_slice, :, :, coil]), cmap='gray', vmin = 0, vmax = 10000)
    #     # plt.imshow(maskB[x_slice, :, :], alpha = 0.4, cmap = 'Reds')
    #     # plt.imshow(brain_mask[x_slice, :, :], alpha = 0.4, cmap = 'Greens')
    #     # plt.imshow(maskA[x_slice, :, :], alpha = 0.4, cmap = 'Greens')
    #     plt.title(f'Virtual Coil {coil+1}')
    # plt.show()

    # exit()

    sir_threshold = float(inputs["sir_threshold"])
    
    i = top_nv_sir(V, nc, A, B, sir_threshold) # finding top nv eigenvectors based on SIR
    top_nv_signal_retained(V, nc, A, B, sir_threshold) # finding top nv eigenvectors based on SIR
    
    # exit()
    
    # V = orth(V)
    print(f'The top nv eigenvectors contain the first {i} eigenvectors')
    V_retain = V[:,:i] # retain only the top nv eigenvectors 
    V_retain = orth(V_retain)
    virtual_coil_data = form_virtual_coil_data(V_retain, data) # forming virtual coils, with eigenvectors as linear combo weights

    # shift, compute fourier transform, shift over x, y, z axes
    # image = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(virtual_coil_data, axes = (0, 1, 2)), axes=(0, 1, 2)), axes = (0, 1, 2))
    image = np.fft.ifftn(virtual_coil_data, axes = (0, 1, 2)) #ATTENTION: ONLY FOR THE CALGARY DATASET, WHICH WAS ALREADY SHIFTED

    image_rss = rsos(image) # find root sum of squares of image over the channel dimension
    image_rss = (image_rss - image_rss.min()) / (image_rss.max() - image_rss.min())

    kspace_rss = rsos(virtual_coil_data) # compute rsos of k-space data

    display(image_rss, kspace_rss, x_slice, y_slice, z_slice, brain_mask, maskB, gap*2) # display images2