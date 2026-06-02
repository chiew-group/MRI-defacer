import numpy as np 
import nibabel as nib 
from totalsegmentator.python_api import totalsegmentator 
import os
# from scipy.ndimage import zoom

GREEN = '\033[92m'
RESET = '\033[0m'

def gen_mask(input_fp, output_fp, saveID):
        '''
        To generate numpy binary masks for the brain and face regions using TotalSegmentator 
        
        Parameters
        ----------
        input_fp -> str: path to nifti file of initial reconstruction
        output_fq -> str: path to output nifti masks of face and brain
        saveID -> str: ID to save numpy masks 
        '''

        # calling TotalSegmentator API for face_mr task
        totalsegmentator(
                        input = input_fp, 
                        output = output_fp,
                        task = "face_mr",
                        device = "gpu"
        )
        # calling TotalSegmentator API for brain task
        totalsegmentator(
                        input = input_fp, 
                        output = output_fp, 
                        task = "total_mr", 
                        roi_subset = ["brain"],
                        fast = True,
                        device = "gpu"
        )

        #face_mask = nib.load(os.path.join(output_fp, "face.nii.gz"))
        #brain_mask = nib.load(os.path.join(output_fp, "brain.nii.gz"))

        # convert to numpy data 
        #face_mask_np = face_mask.get_fdata()
        #brain_mask_np = brain_mask.get_fdata()
        
        # save as numpy masks
        #np.save(f'segmentations/face_mask_{saveID}.npy', face_mask_np)
        #np.save(f'segmentations/brain_mask_{saveID}.npy', brain_mask_np)

        print(GREEN + "Mask saved successfully" + RESET)
