import numpy as np
import nibabel as nib
import os
import subprocess
import shutil
import ants
import antspynet

GREEN = '\033[92m'
RESET = '\033[0m'


def gen_mask(input_fp, output_fp, saveID):
        '''
        To generate numpy binary masks for the brain and face regions using GRACE CLI. 
        
        Parameters
        ----------
        input_fp -> str: path to nifti file of initial reconstruction
        output_fq -> str: path to output nifti masks of face and brain
        saveID -> str: ID to save numpy masks 
        '''

        input = ants.image_read(input_fp) # load input data

        whole_head = ants.get_mask(input, cleanup = 1) # get mask of entire head
        # ants.image_write(whole_head, 'segmentations/ant_whole_head.nii.gz') # save whole head data

        brain_probability_mask = antspynet.utilities.brain_extraction(image=input, modality="t1")
        brain_bin_mask = ants.threshold_image(brain_probability_mask, low_thresh=0.5, high_thresh=1.0, binary=True)
        ants.image_write(brain_bin_mask, os.path.join(output_fp, 'brain.nii.gz'))

        whole_head_np = whole_head.numpy().astype(bool)
        brain_bin_mask_np = brain_bin_mask.numpy().astype(bool)

        face_np = whole_head_np & ~brain_bin_mask_np # logical AND NOT

        face = ants.from_numpy(face_np.astype(np.uint8), 
                        origin=whole_head.origin, 
                        spacing=whole_head.spacing, 
                        direction=whole_head.direction)
        ants.image_write(face, os.path.join(output_fp, 'face.nii.gz'))

        print(GREEN + "Mask saved successfully" + RESET)