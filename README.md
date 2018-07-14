Pixelartor ¯\_(ツ)_/¯

usage: main.py [-h] [--colors COLORS] [--right_pecrentile RIGHT_PECRENTILE]
               [--left_pecrentile LEFT_PECRENTILE]
               input_img output_img

positional arguments:
  input_img             Input image.
  output_img            Output image.

optional arguments:
  -h, --help            show this help message and exit
  --colors COLORS, -c COLORS
                        Number of colors per channel. 8 as default.
  --right_pecrentile RIGHT_PECRENTILE, -r RIGHT_PECRENTILE
                        Contrast stretching, right percentile, 98 as default.
                        Int in range [left percentile..100]
  --left_pecrentile LEFT_PECRENTILE, -l LEFT_PECRENTILE
                        Contrast stretching, left percentile, 2 as default.
                        Int in range [0..right_percentile]
