import cv2
import mmcv
import numpy as np
import torch

from mmdet.apis import init_detector, inference_detector

score_thr = 0.8
video_reader = mmcv.VideoReader("dataset/test_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(
    './color_splash.mp4', fourcc, video_reader.fps,
    (video_reader.width, video_reader.height))
print(len(video_reader))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = init_detector('new_mrcnn/mask_rcnn_r50_fpn_1x_ballon.py', 'new_mrcnn/latest.pth', device=device)

for frame in video_reader:
    result = inference_detector(model, frame)
    mask = None
    masks = result[1][0]
    for i in range(len(masks)):
        if result[0][0][i][-1] >= score_thr:
            if not mask is None:
                mask = mask | masks[i]
            else:
                mask = masks[i]

    # 获取各通道mask像素

    masked_b = frame[:, :, 0] * mask
    masked_g = frame[:, :, 1] * mask
    masked_r = frame[:, :, 2] * mask
    masked = np.concatenate([masked_b[:,:,None],masked_g[:,:,None],masked_r[:,:,None]],axis=2)

    # frame转灰度图

    un_mask = 1 - mask
    frame_b = frame[:, :, 0] * un_mask
    frame_g = frame[:, :, 1] * un_mask
    frame_r = frame[:, :, 2] * un_mask
    frame = np.concatenate([frame_b[:, :, None], frame_g[:, :, None], frame_r[:, :, None]], axis=2).astype(np.uint8)
    frame = mmcv.bgr2gray(frame, keepdim=True)
    frame = np.concatenate([frame, frame, frame], axis=2)
    # mask加到灰度图中

    frame += masked

    # frame = model.show_result(frame, result, score_thr=score_thr)
    video_writer.write(frame)

#   掩码计算->灰度图->灰度图*3->替换值

video_writer.release()
cv2.destroyAllWindows()