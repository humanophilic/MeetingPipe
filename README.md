# MeetingPipe
Quantifying multi-person meetings based on multi-modal micro-behavior analysis.

# Getting Started
The code was tested on Ubuntu 20.04 with CUDA 11.1 and cuDNN 8.1.0.
## Dependencies
Start from building a new conda environment.
```
conda create -n MeeingPipe python=3.7.11
conda activate MeeingPipe
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirements.txt
conda update ffmpeg
```
## Pretrained models
You can download the pretrained model from [here](https://drive.google.com/drive/folders/1-8VPqTqPFTBHVG_A2Ji_nH0nrOy4r7_e?usp=sharing):
1. Download face track model and place it into `face_track/faceDetector/s3fd/sfd_face.pth`.
2. Download Active Speaker Detection(ASD) model and place it into `speak/pretrain_TalkSet.model`
3. Download Head Pose Estimation(HPE) model and place it into `nod/6DRepNet_300W_LP_AFLW2000.pth.`
## Data preparation
We build an end-to-end script that detects and recognizes three indicators, speaking, nodding and smile, from the input video.  
You can place the raw video file (any video format that OpenCV supports: mp4, avi, etc.) into the `demo` folder.
## Usage
```
python pipeline.py --videoPath /path/to/video --nGuest <number of faces in the video> --gpus <number of gpus used>
```
Example:  
```
python pipeline.py --videoPath demo/demo_video.mov --nGuest 1 --gpus 2
```
The operation time depends on the video duration, number of faces, number of gpus, etc.  
You can find the results in the `demo` folder, which consists of two parts:  
1. Result visualization video: `demo/result/<your video name>/pyavi/output.mp4`
2. Result csv file: `demo/csv/<your video name>/scores.csv`  
Which contains scores of speak(>0), non-speak(<0), nod(=1), non-nod(=0), smile(=1), non-smile(=0)

It is not recommended to use cpu only, which is quite time-consuming and you need to modify all `cuda` in the script into `cpu`.

# Acknowledge
We study several useful project during our coding process, you can reference from:
1. Active speaker detection module is learnt from this [repository](https://github.com/TaoRuijie/TalkNet-ASD).
2. Head pose estimation module is learnt from this [repository](https://github.com/thohemp/6DRepNet).

# Contact us
For any questions or requests, please contact us at chen.chenhao.419@s.kyushu-u.ac.jp
