import mmcv
from mmdet.apis import init_detector, inference_detector
import cv2
import argparse

import skvideo
skvideo.setFFmpegPath("ffmpeg-git-20220910-amd64-static")
import skvideo.io
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="mask_rcnn_30/mask_rcnn_r50_fpn_fp16_1x_balloon.py", type=str)
    parser.add_argument("--checkpoint_file", default='mask_rcnn_30/latest.pth', type=str)
    # parser.add_argument("--n_gpus", type=int, default=1)
    # parser.add_argument("--infer_list", default="voc12/train.txt", type=str)

    args = parser.parse_args()
    return args



def main(args):
    my_video=mmcv.VideoReader('dataset/test_video.mp4')
    # my_img='dataset/balloon/val/5603212091_2dfe16ea72_b.jpg'
    # config_file='mask_rcnn_balloon_30/mask_rcnn_r50_fpn_fp16_1x_balloon.py'
    # checkpoint_file='mask_rcnn_balloon_30/epoch_30.pth'
    model = init_detector(args.config_file, args.checkpoint_file, device='cuda:0')
    video_writer=skvideo.io.FFmpegWriter("outputvideo.mp4",outputdict={'-pix_fmt': 'yuv420p'})
    # inference_detector(model, my_img,out_file='result111.jpg')
    output_img_all=[]
    for frame in tqdm(my_video):
        result = inference_detector(model, frame)
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        output_img=model.show_result(frame_gray, result)
        video_writer.writeFrame(output_img)
        # output_img_all.append(output_img)
    video_writer.close()
    print('done')
    # output_img_result=np.stack(output_img_all,axis=0)
    # i,_,_,_=output_img_result.shape
    # for j in range(i):
    #     video_writer.writeFrame(output_img_result[j, :, :, :])
    # video_writer.close()

if __name__=='__main__':
    args = parse_args()
    main(args)