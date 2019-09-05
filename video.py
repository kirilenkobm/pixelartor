#!/usr/bin/env python3
"""Pixelate video."""
import argparse
from skimage import io
from skimage import transform as tf
from skimage import exposure
from skimage import feature
from skimage import filters
import numpy as np
import sys
import moviepy.editor as mp
from main import pixel

try:
    in_video = sys.argv[1]
    out_video = sys.argv[2]
except IndexError:
    sys.stderr.write("Usage: {0} [in_video] [out_video]\n".format(sys.argv[0]))
    sys.exit(0)


vidos = mp.VideoFileClip(filename=in_video, audio=False).subclip(5, 8)
clip_pixel = vidos.fl_image(pixel)
clip_pixel.write_videofile(out_video)
