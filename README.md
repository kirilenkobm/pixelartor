# Pixelartor

A tool to produce pixel-art.

## Usage

First, install python libraries listed in the requirements.txt

```shell
pip3 install -r requirements.txt
```

For video processing you also need moviepy library.
This direction was not tested and developed properly yet.

Then call main.py:

```txt
usage: main.py [-h] [--right_pecrentile RIGHT_PECRENTILE]
               [--left_pecrentile LEFT_PECRENTILE] [--magnify] [--interlacing]
               [--edges_sigma EDGES_SIGMA]
               [--egdes_blur_sigma EGDES_BLUR_SIGMA]
               input_img output_img

positional arguments:
  input_img             Input image.
  output_img            Output image.

optional arguments:
  -h, --help            show this help message and exit
  --right_pecrentile RIGHT_PECRENTILE, -r RIGHT_PECRENTILE
                        Contrast stretching, right percentile, 98 as default.
                        Int in range [left percentile..100]
  --left_pecrentile LEFT_PECRENTILE, -l LEFT_PECRENTILE
                        Contrast stretching, left percentile, 4 as default.
                        Int in range [0..right_percentile]
  --magnify, -m         Increase size of the pixels.
  --interlacing, -i     Apply interlacing.
  --edges_sigma EDGES_SIGMA, -e EDGES_SIGMA
                        Sigma for canny filter.
  --egdes_blur_sigma EGDES_BLUR_SIGMA, -s EGDES_BLUR_SIGMA
                        Gaussian filter sigma.
```

Default parameters are fine in most cases.

## What is it for

Before & After

![alt text 1](examples/before_after_1.jpg  "Before 1")

![alt text 1](examples/before_after_2.jpg  "Before 1")

## Examples

![alt text](examples/gal_1.jpg  "Gallery 1")

![alt text](examples/gal_2.jpg  "Gallery 2")

![alt text](examples/gal_3.jpg  "Gallery 3")

![alt text](examples/gal_4.jpg  "Gallery 4")

![alt text](examples/gal_5.jpg  "Gallery 4")

![alt text](examples/gal_6.jpg  "Gallery 4")
