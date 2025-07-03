# MRI Defacer
(name of tool) is a defacing tool for anonymizing MRI data. It employs TotalSegmentator's face_mr task [2] to automatically generate a mask for the facial region, then uses region-optmized virtual coils (ROVir) [1] to deface the MRI data in k-space.

work in progress...!

## Installation and Usage 
Set up a Python environment for (name of tool)

```
conda create -n MRIDEFACER python=3.13
```

Install TotalSegmentator and other required dependencies

```
pip install TotalSegmentator 
pip install matplotlib
pip install scipy
```

Clone this GitHub repository

```
git clone https://github.com/chiew-group/MRI-defacer.git
cd MRI-defacer
```
Edit the config.json file to specify the input data path, slice information, and SIR threshold. (name of tool) excepts raw MRI data in the MRD data format. Run main.py to get defaced images. 

## Citations 
[1] Kim D, Cauley SF, Nayak KS, Leahy RM, Haldar JP. Region- optimized virtual (ROVir) coils: Localization and/or suppression of 
spatial regions using sensor- domain beamforming. Magn Reson Med. 2021;86:197–212. 
https://doi.org/10.1002/mrm.28706 

[2] Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., Segeroth, M., 2023. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence. https://doi.org/10.1148/ryai.230024