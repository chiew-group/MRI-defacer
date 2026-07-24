import numpy as np 
import nibabel as nib 

GREEN = '\033[92m'
RESET = '\033[0m'

def niftify(image, a11, a22, a33, save_path):

    '''
    To convert numpy files into nifti files for image visualization tools and segmentation tools. 

    Parameters
    ----------
        image -> np.ndarray: the image space data with shape (x, y, z, ch)
        a11 -> float: voxel spacing along x axis
        a22 -> float: voxel spacing along y axis
        a33 -> float: voxel spacing along z axis
        name -> str: the ID of the nifti to be saved  

    '''
    
    # normalize intensity range to 0-255
    image = (image - image.min()) / (image.max() - image.min()) * 255

    # converting numpy data to nifti file 
    affine = np.array([[a11, 0.0, 0.0, -image.shape[0]//2], 
                    [0.0, a22, 0.0, -image.shape[1]//2],
                    [0.0, 0.0, a33, -image.shape[2]//2],
                    [0.0, 0.0, 0.0, 1.0]
                    
    ]) 
    nifti_img = nib.Nifti1Image(image.astype(np.uint8), affine)
    # print(nifti_img.affine)

    # saving the nifti file 
    fp = f'{save_path}.nii.gz'
    nib.save(nifti_img, fp)
    print(GREEN + f"[UPDATE] Nifti file successfully saved to {fp}" + RESET)