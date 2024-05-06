# 从视频中抽出一帧
# ffmpeg -i input.mp4 -vf "fps=1/60" output_%03d.png

# 视频中将人脸区域裁剪出来
# ffmpeg -i sn_test.mp4 -filter:v "crop=453:388:315:11" sn_test_face.mp4
