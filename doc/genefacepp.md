[📘使用文档]() |
[🛠安装教程]() |
[👀模型库]() |
[🆕更新日志]() |
[🚀进行中的项目]() |
[🤔报告问题]()


**数据处理**

```
# 从视频中抽出一帧
ffmpeg -i input.mp4 -vf "fps=1/60" output_%03d.png

# 视频中将人脸区域裁剪出来
ffmpeg -i sn_test.mp4 -filter:v "crop=453:388:315:11" sn_test_face.mp4

# 步骤0. 将视频Crop到512x512分辨率，25FPS，确保每一帧都有目标人脸
export PYTHONPATH=./
export VIDEO_ID=sn429_face
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 data/raw/videos/${VIDEO_ID}_512.mp4
mv data/raw/videos/${VIDEO_ID}.mp4 data/raw/videos/${VIDEO_ID}_to_rm.mp4
mv data/raw/videos/${VIDEO_ID}_512.mp4 data/raw/videos/${VIDEO_ID}.mp4

# 步骤1: 提取音频特征, 如mel, f0, hubuert
mkdir -p data/processed/videos/${VIDEO_ID}
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -f wav -ar 16000 data/processed/videos/${VIDEO_ID}/aud.wav 
python data_gen/utils/process_audio/extract_hubert.py --video_id=${VIDEO_ID}
python data_gen/utils/process_audio/extract_mel_f0.py --video_id=${VIDEO_ID

# 步骤2. 提取图片
mkdir -p data/processed/videos/${VIDEO_ID}/gt_imgs
ffmpeg -i data/raw/videos/${VIDEO_ID}.mp4 -vf fps=25,scale=w=512:h=512 -qmin 1 -q:v 1 -start_number 0 data/processed/videos/${VIDEO_ID}/gt_imgs/%08d.jpg
python data_gen/utils/process_video/extract_segment_imgs.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 # extract image, segmap, and background

步骤3. 提取lm2d_mediapipe
提取2D landmark用于之后Fit 3DMM
num_workers是本机上的CPU worker数量；total_process是使用的机器数；process_id是本机的编号
python data_gen/utils/process_video/extract_lm2d.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4

步骤3. Fit 3DMM
python data_gen/utils/process_video/fit_3dmm_landmark.py --ds_name=nerf --vid_dir=data/raw/videos/${VIDEO_ID}.mp4 --reset  --debug --id_mode=global

步骤4. Binarize（将数据打包）
python data_gen/runs/binarizer_nerf.py --video_id=${VIDEO_ID}
```

**训练**

```
# 训练 Head NeRF 模型
# 模型与tensorboard会被保存在 `checkpoints/<exp_name>`
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/May/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/may_head --reset

# 训练 Torso NeRF 模型
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config=egs/datasets/May/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/may_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/may_head --reset

```

**推理**

```
# 使用我们提供的推理脚本.
CUDA_VISIBLE_DEVICES=0  python inference/genefacepp_infer.py --head_ckpt= --torso_ckpt=/home/lgd/common/GeneFacePlusPlus/checkpoints/motion2video_nerf/may_torso/ --drv_aud=/home/lgd/common/GeneFacePlusPlus/audio_0609.wav

# --debug 选项可以可视化一些中间过程与特征
CUDA_VISIBLE_DEVICES=0  python inference/genefacepp_infer.py --head_ckpt= -torso_ckpt=/home/lgd/common/GeneFacePlusPlus/checkpoints/motion2video_nerf/may_torso/ --drv_aud=/home/lgd/common/GeneFacePlusPlus/audio_0609.wav --debug
```    