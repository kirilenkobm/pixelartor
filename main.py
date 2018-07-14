"""Entry point."""
import argparse
from skimage import io
from skimage import transform as tf
from skimage import exposure
import numpy as np
import sys

# const
W, H = 800, 500
W_pix = 160
H_pix = 100


def parse_args():
    """Define and parse args."""
    app = argparse.ArgumentParser()
    app.add_argument("input_img", type=str, help="Input image.")
    app.add_argument("output_img", type=str, help="Output image.")
    app.add_argument("--colors", "-c", type=int, default=8,
                     help="Number of colors per channel. 8 as default.")
    app.add_argument("--right_pecrentile", "-r", type=int, default=98,
                     help="Contrast stretching, right percentile, 98 as default. "
                          "Int in range [left percentile..100]")
    app.add_argument("--left_pecrentile", "-l", type=int, default=2,
                     help="Contrast stretching, left percentile, 2 as default. "
                          "Int in range [0..right_percentile]")

    if len(sys.argv) < 2:  # no arguments, show help in this case
        app.print_help()
        sys.exit(0)
    args = app.parse_args()
    return args


def main():
    """Main func."""
    args = parse_args()
    # im is an W x H x 3 array of 0..1
    im = tf.resize(io.imread(args.input_img), (H, W))
    pic = np.zeros((H, W, 3))
    # starts / steps
    w_starts = np.linspace(0, W, endpoint=False, num=W_pix)
    w_step = w_starts[1] - w_starts[0]
    h_starts = np.linspace(0, H, endpoint=False, num=H_pix)
    h_step = h_starts[1] - h_starts[0]
    # color bins
    bins = np.linspace(0.0, 1.0, num=args.colors)
    # contrast percentiles
    perc_left, perc_right = np.percentile(im, (args.left_pecrentile, args.right_pecrentile))
    im = exposure.rescale_intensity(im, in_range=(perc_left, perc_right))

    # main loop \ "pixel" by "pixel"
    for w_startf in w_starts:
        for h_startf in h_starts:
            # define the coords of the square
            w_start = int(w_startf)
            h_start = int(h_startf)
            w_end = int(w_start + w_step)
            h_end = int(h_start + h_step)

            # ... and the color of this "pixel"
            rot_mean = im[h_start: h_end, w_start: w_end, 0].mean()
            grun_mean = im[h_start: h_end, w_start: w_end, 1].mean()
            blau_mean = im[h_start: h_end, w_start: w_end, 2].mean()
            digitized = np.digitize([rot_mean, grun_mean, blau_mean], bins=bins)

            pic[h_start: h_end, w_start: w_end, 0] = bins[digitized[0] - 1]
            pic[h_start: h_end, w_start: w_end, 1] = bins[digitized[1] - 1]
            pic[h_start: h_end, w_start: w_end, 2] = bins[digitized[2] - 1]

    # save result
    io.imsave(args.output_img, pic)


if __name__ == "__main__":
    main()
