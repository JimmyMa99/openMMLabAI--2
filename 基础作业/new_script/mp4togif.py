from moviepy.editor import *

clip = (VideoFileClip("color_splash.mp4"))
clip.write_gif("color_splash.gif")
