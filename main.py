#!/usr/bin/env python3
"""Pixelartor. Standalone tool."""
import argparse
from skimage import io
from skimage import transform as tf
from skimage import exposure
from skimage import feature
from skimage import filters
import numpy as np
import sys

# const
W, H = 800, 500
four_bits = np.linspace(0.0, 1.0, num=9)
two_bits = np.linspace(0.0, 1.0, num=5)
GR_THR = 1.04
sys.setrecursionlimit(50000)


class ImageGraph:
    """Image-based graph."""
    def __init__(self, arr, w, h, mapped):
        self.arr = arr
        self.mapped = mapped
        self.start_w = w
        self.start_h = h
        self.w_border = arr.shape[0]
        self.h_border = arr.shape[1]
        self.parentship = {}
        self.checked = {(w, h)}
        self.grad = {(w, h)}

    def get_moves(self, w, h):
        """Return all possible moves for w and h given."""
        moves = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if i == j == 0:
                    continue  # points to itself
                if w + i >= self.w_border or w + i < 0:
                    continue  # out of array
                elif h + j >= self.h_border or h + j< 0:
                    continue  # out of array
                elif (w + i, h + j) in self.mapped:
                    continue  # checked in prev generation
                elif (w + i, h + j) in self.checked:
                    continue  # checked in this generation
                # elif (w + i, h + j) == self.parentship.get((w, h)):
                #     continue  # point to the parent
                else:
                    moves.append((w + i, h + j))
        return moves

    def recurs(self, w, h):
        """Recursively go through the image."""
        self.checked.add((w, h))
        moves = self.get_moves(w, h)
        color_init = self.arr[w, h]
        for move in moves:
            self.parentship[move] = (w, h)
            self.checked.add(move)
            color_move = self.arr[move[0], move[1]]
            if is_grad(color_init, color_move):
                # if grad - check next vertex
                self.grad.add(move)
                self.recurs(move[0], move[1])
        return

    def get_mapped(self):
        """Return mapped vertexes."""
        return self.grad


def parse_args():
    """Define and parse args."""
    app = argparse.ArgumentParser()
    app.add_argument("input_img", type=str, help="Input image.")
    app.add_argument("output_img", type=str, help="Output image.")
    app.add_argument("--colors", "-c", type=int, default=8,
                     help="Number of colors per channel. 8 as default.")
    app.add_argument("--right_pecrentile", "-r", type=int, default=98,
                     help="Contrast stretching, right percentile, 90 as default. "
                          "Int in range [left percentile..100]")
    app.add_argument("--left_pecrentile", "-l", type=int, default=4,
                     help="Contrast stretching, left percentile, 2 as default. "
                          "Int in range [0..right_percentile]")
    app.add_argument("--magnify", "-m", action="store_true", dest="magnify",
                     help="Increase size of the pixels.")
    app.add_argument("--interlacing", "-i", action="store_true", dest="interlacing",
                     help="Apply interlacing.")
    app.add_argument("--edges_sigma", "-e", type=float, default=3.0,
                     help="Sigma for canny filter.")
    app.add_argument("--egdes_blur_sigma", "-s", type=float, default=0.3,
                     help="Gaussian filter sigma.")
    if len(sys.argv) < 2:  # no arguments, show help in this case
        app.print_help()
        sys.exit(0)
    args = app.parse_args()
    return args


def eprint(msg, end="\n"):
    """Print for stderr."""
    sys.stderr.write(msg + end)


def make_layout(args):
    """Define coordinates of blocks."""
    # define start points / step size
    W_pix = 160 if not args.magnify else 80
    H_pix = 100 if not args.magnify else 50
    w_starts = np.linspace(0, W, endpoint=False, num=W_pix)
    w_step = w_starts[1] - w_starts[0]
    h_starts = np.linspace(0, H, endpoint=False, num=H_pix)
    h_step = h_starts[1] - h_starts[0]
    # color bins
    bins = np.linspace(0.0, 1.0, num=args.colors)
    return W_pix, H_pix, w_starts, w_step, h_starts, h_step, bins


def enhance_edges(im, args):
    """Paint edges black."""
    edges = feature.canny(im[:, :, 1], sigma=args.edges_sigma)
    edges = filters.gaussian(edges, sigma=args.egdes_blur_sigma)
    edges = edges.astype(float) * 2.8
    im[:, :, 0] -= edges
    im[:, :, 1] -= edges
    im[:, :, 2] -= edges
    im[im < 0.0] = 0.0
    return im


def convert_to_256(red, green, blue):
    """Convert 0..1 values to R3G3B2 palette."""
    # extract indexes
    red_index = np.digitize(red, bins=four_bits)
    green_index = np.digitize(green, bins=four_bits)
    blue_index = np.digitize(blue, bins=two_bits)
    # get colors | compensate left-sided
    # print(red_index, green_index, blue_index)
    red_color = four_bits[red_index - 1]
    green_color = four_bits[green_index - 1]
    blue_color = two_bits[blue_index - 1]
    # print(red_color, green_color, blue_color)
    return (red_color, green_color, blue_color)


def is_grad(color_1, color_2):
    """Return true if colors arent so different."""
    for i in range(3):
        chan = (color_1[i], color_2[i])
        if not max(chan) <= min(chan) * GR_THR:
            return False
    return True


def detect_gradients(im, w_starts, h_starts, w_step, h_step):
    """Make gradient map for the image."""
    grad_map = np.zeros((len(w_starts), len(h_starts)))
    ave_colors = np.zeros((len(w_starts), len(h_starts), 3))
    # extract colors for each node
    for w_num, w_startf in enumerate(w_starts):
        for h_num, h_startf in enumerate(h_starts):
             # define the coords of the square
            w_start, h_start = int(w_startf), int(h_startf)
            w_end, h_end = int(w_start + w_step), int(h_start + h_step)
            rot_mean = im[h_start: h_end, w_start: w_end, 0].mean()
            grun_mean = im[h_start: h_end, w_start: w_end, 1].mean()
            blau_mean = im[h_start: h_end, w_start: w_end, 2].mean()
            ave_colors[w_num][h_num] = [rot_mean, grun_mean, blau_mean]

    # init values
    mapped = set()

    # start graph
    w_size, h_size = len(w_starts), len(h_starts)
    eprint("Find gradiends...")
    cluster = 0
    for w in range(w_size):
        for h in range(h_size):
            if (w, h) in mapped:  # we checked it previously --> skip
                continue
            mapped.add((w, h))
            ig = ImageGraph(ave_colors, w, h, mapped)
            ig.recurs(w, h)
            new_mapped = ig.get_mapped()
            mapped = mapped | new_mapped
            if len(new_mapped) == 1:
                continue
            cluster += 1
            for elem in new_mapped:
                grad_map[elem[0], elem[1]] = cluster
        eprint("Row {0}/{1}".format(w, w_size), end="\r")

    return ave_colors, grad_map, cluster


def colors_average(colors):
    """Return average color for a set of colors."""
    group_size = len(colors)
    red_sum, green_sum, blue_sum = 0, 0, 0
    for color in colors:
        red_sum += color[0]
        green_sum += color[1]
        blue_sum += color[2]
    red_ave, green_ave, blue_ave = red_sum / group_size, green_sum / group_size, blue_sum / group_size
    return [red_ave, green_ave, blue_ave]


def main():
    """Main func."""
    args = parse_args()
    # im is an W x H x 3 array of 0..1
    im = tf.resize(io.imread(args.input_img), (H, W))
    eprint("Precomputing pixels, preprocessing the image...")
    pic = np.zeros((H, W, 3))  # init empty image with zeros
    # image markup
    W_pix, H_pix, w_starts, w_step, h_starts, h_step, _ = make_layout(args)
    # contrast percentiles
    perc_left, perc_right = np.percentile(im, (args.left_pecrentile, args.right_pecrentile))
    im = exposure.rescale_intensity(im, in_range=(perc_left, perc_right))
    # find edges
    im = enhance_edges(im, args)
    ave_colors, grad_map, clusters_num = detect_gradients(im, w_starts, h_starts, w_step, h_step)
    for clust_num in range(1, clusters_num + 1):
        colors = ave_colors[grad_map == clust_num]
        cluster_ave = colors_average(colors)
        ave_colors[grad_map == clust_num] = cluster_ave

    # main loop \ "pixel" by "pixel"
    eprint("Assign colors...")
    for w_num, w_startf in enumerate(w_starts):
        for h_num, h_startf in enumerate(h_starts):
            darker = 0.6 if h_num % 2 == 0 and args.interlacing else 1.0
            # define the coords of the square
            w_start = int(w_startf)
            h_start = int(h_startf)
            w_end = int(w_start + w_step)
            h_end = int(h_start + h_step)

            # ... and the color of this "pixel"
            # rot_mean = im[h_start: h_end, w_start: w_end, 0].mean()
            # grun_mean = im[h_start: h_end, w_start: w_end, 1].mean()
            # blau_mean = im[h_start: h_end, w_start: w_end, 2].mean()
            # digitized = np.digitize([rot_mean, grun_mean, blau_mean], bins=bins)
            color = ave_colors[w_num, h_num]
            digitized = convert_to_256(color[0], color[1], color[2])
            # pic[h_start: h_end, w_start: w_end, 0] = bins[digitized[0] - 1] * darker
            # pic[h_start: h_end, w_start: w_end, 1] = bins[digitized[1] - 1] * darker
            # pic[h_start: h_end, w_start: w_end, 2] = bins[digitized[2] - 1] * darker
            pic[h_start: h_end, w_start: w_end, 0] = digitized[0] * darker
            pic[h_start: h_end, w_start: w_end, 1] = digitized[1] * darker
            pic[h_start: h_end, w_start: w_end, 2] = digitized[2] * darker

    # save result
    io.imsave(args.output_img, pic)
    eprint("Done")
    sys.exit(0)


if __name__ == "__main__":
    main()
