import numpy as np 
import nibabel as nib 

GREEN = '\033[92m'
RESET = '\033[0m'

def niftify(image, a11, a22, a33, name):
    
    image = (image - image.min()) / (image.max() - image.min()) * 255

    # converting numpy data to nifti file 
    affine = np.array([[a11, 0.0, 0.0, -image.shape[0]//2], 
                    [0.0, a22, 0.0, -image.shape[1]//2],
                    [0.0, 0.0, a33, -image.shape[2]//2],
                    [0.0, 0.0, 0.0, 1.0]
                    
    ]) 
    nifti_img = nib.Nifti1Image(image.astype(np.uint8), affine)
    print(nifti_img.affine)

    fp = f'results/{name}.nii'
    nib.save(nifti_img, fp)
    print(GREEN + "Nifti file successfully saved" + RESET)
