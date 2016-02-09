# diff_filter_emu

Add pseudo diffusion filter effect to given photo image.

## Description

diff_filter_emu is command line tool that generates image with pseudo diffusion filter effect, this effect is realized with cascading (adding) gradual gaussian filtered layers of original image.
This program is especially good for starry night photography enhancement, however, can be used for other kinds of photography.
Working with large image size (such as more than FullHD size) is recommended.

Before

<img src="https://raw.githubusercontent.com/delphinus1024/diff_filter_emu/master/before.png" style="width: 600px;"/>

After (w/ default parameters)

<img src="https://raw.githubusercontent.com/delphinus1024/diff_filter_emu/master/after.png" style="width: 600px;"/>

## Features

- Generates image with diffusion filter effect.
- Some values can be parameterized to control the degree of strength of the effect.

## Requirement

- OpenCV 3.0.0 or above is preferable.
- Checked with win7 32bit + msys2 + gcc

## Usage

1. Type following command in your command window.
2. diff_filter_emu -i=input_image [-o=output_image] [-n=iteration] [-d=decay_factor] [-f=decay_offset] [-g=gamma] [-s]
3. [] is optional.

- -i=input_image:    Original image file.
- -o=output_image:   Output image file. (default: result.tif)
- -n=iteration:      Number of gradual gaussian filtered layers. (default: 5)
- -d=decay_factor:   Gain decay of layers. the more values, the more decay. (default: 5.0)
- -f=decay_offset:   Adjust decay of the first layer. If set to 0, gain of the first layer is set to 1 (means full opacity). (default: 0.1)
- -g=gamma:          Gamma correction factor. correction is done before processing and reverse gamma corrention is done after processing.  Larger gamma value gives larger effect to bright area. (default: 1.3)
- -s:              Add this option when show the original and processed image when finished.
	
## Installation

1. Modify Makefile according to your OpenCV inludes and libs environment.
2. make

## Author

delphinus1024

## License

[MIT](https://raw.githubusercontent.com/delphinus1024/diff_filter_emu/master/LICENSE.txt)

