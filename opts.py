import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    # main
    parser.add_argument('--videoPath', type=str, default='demo/demo_video.mov', help='Demo video path')
    parser.add_argument('--nGuest', type=int, default=1, help='Number of participants')
    parser.add_argument('--gpus', type=int, default=2, help='Number of gpus')

    parser.add_argument('--cropDir', type=str, default='demo/result', help='Path for cropped video')
    parser.add_argument('--csvDir', type=str, default='demo/csv', help='Path for csv results')

    # face track
    parser.add_argument('--facedetScale', type=float, default=0.25, help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    parser.add_argument('--minTrack', type=int, default=10, help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet', type=int, default=10, help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
    parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')
    parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of workers for face track')
    parser.add_argument('--start', type=int, default=0, help='The start time of the video')
    parser.add_argument('--duration', type=int, default=0, help='The duration of face track in the video, when set as 0, will extract the whole video')

    # speak detection
    parser.add_argument('--speakModel', type=str, default='./speak/pretrain_TalkSet.model', help='Path for the pretrained speak model')

    # nod detection
    parser.add_argument('--nodModel', type=str, default='./nod/ckpt/epoch_150.pth', help='Path for the pretrained nod model')
    parser.add_argument('--snapshot', type=str, default='./nod/6DRepNet_300W_LP_AFLW2000.pth', help='6DRepNet snapshot path')

    # smile detection
    parser.add_argument('--smileModel', type=str, default='./smile/smilenet_epoch_4.pth', help='Path for the pretrained smile model')

    args, unparsed = parser.parse_known_args()

    return args
