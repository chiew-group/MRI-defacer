import numpy as np
import os
import ants
import antspynet

GREEN = '\033[92m'
RESET = '\033[0m'


def gen_mask(input_fp, output_fp, modality):
        '''
        To generate numpy binary masks for the brain and face regions using ANTsPY and ANTsPyNet.
        
        Parameters
        ----------
        input_fp -> str: path to nifti file of initial reconstruction
        output_fq -> str: path to output nifti masks of face and brain
        saveID -> str: ID to save numpy masks 
        '''

        input = ants.image_read(input_fp) # load input data

        whole_head = ants.get_mask(input, cleanup = 1) # get mask of entire head

        os.makedirs(output_fp, exist_ok=True) # make an output directory to store outputs

        brain_probability_mask = antspynet.utilities.brain_extraction(image=input, modality=modality) # get probability mask for brain
        brain_bin_mask = ants.threshold_image(brain_probability_mask, low_thresh=0.5, high_thresh=1.0, binary=True) # get binary mask for brain
        ants.image_write(brain_bin_mask, os.path.join(output_fp, 'brain.nii.gz')) # save nifti of binary brain mask
        
        whole_head_np = whole_head.numpy().astype(bool) 
        brain_bin_mask_np = brain_bin_mask.numpy().astype(bool)

        # lobes = antspynet.desikan_killiany_tourville_labeling(t1=input, do_preprocessing=True, return_probability_images=False,
        #                                                              do_lobar_parcellation=True, version=0, verbose=False)
        
        # frontal_lobe_mask = ((lobes['lobar_parcellation']).numpy() == 1).astype(np.uint8)
        # ants.image_write(frontal_lobe_mask, os.path.join(output_fp, 'frontal_lobe.nii.gz'))

        face_np = whole_head_np & ~brain_bin_mask_np # logical AND NOT to find the face mask

        face = ants.from_numpy(face_np.astype(np.uint8), 
                        origin=whole_head.origin, 
                        spacing=whole_head.spacing, 
                        direction=whole_head.direction)
        ants.image_write(face, os.path.join(output_fp, 'face.nii.gz')) # save nifti of binary face mask
 
        print(GREEN + "Mask saved successfully" + RESET)