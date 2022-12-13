from moviepy.editor import *

video_name = "demovideo.avi"
clip = VideoFileClip(video_name)
video = concatenate_videoclips([clip])
video.write_videofile(video_name.split('.')[0] + ".mp4", fps=30, codec='libx264', audio=False, preset="medium")