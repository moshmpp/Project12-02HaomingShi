# Simplified from https://github.com/IanMJB/COMP6206-Invariant-Fourier-Descriptors/blob/master/src/fourier_toolbox.py

# Future imports.
from __future__ import division

# External libraries.
import numpy as np

class fourier_toolbox(object):

    # Turns the contours provided by OpenCV into a numpy complex array.
    def contour_to_complex(self, contours, layer = 0):
        # [layer] strips out the array we care about.
        # Advanced indexing in numpy: [:, 0]
        # : gets ALL 'rows'.
        # 0 grabs the first element in each 'row'.
        contour    = contours[layer][:, 0]

        # Creates an empty np struct.
        # shape gives (len, 1, 2), i.e. an array of pairs length len.
        # [:-1] gives an array of elements length len.
        contour_complex            = np.empty(contour.shape[:-1], dtype = complex)
        contour_complex.real    = contour[:, 0]
        # Negated as OpenCV flips the y-axes normally, eases visualisation.
        contour_complex.imag    = -contour[:, 1]

        return contour_complex

    # Gets the lowest X% of frequency values from the fourier values.
    # Places back into the correct order.
    def get_low_frequencies_percentage(self, fourier_val, percent_to_keep):
        to_get        = int(len(fourier_val) * (percent_to_keep / 100))

        return self.get_low_frequencies(fourier_val, to_get)

    # Gets the lowest X of frequency values from the fourier values.
    # Places back into the correct order.
    def get_low_frequencies(self, fourier_val, to_get):
        fourier_freq        = np.fft.fftfreq(len(fourier_val))

        frequency_indices    = []
        for index, val in enumerate(fourier_freq):
            frequency_indices.append([index, val])

        # Sorts on absolute value of frequency (want negative and positive).
        frequency_indices.sort(key = lambda tuple: abs(tuple[1]))

        raw_values    = []
        for i in range(0, to_get):
            index    = frequency_indices[i][0]
            raw_values.append([fourier_val[index], index])

        # Sort based on original ordering.
        raw_values.sort(key = lambda tuple: tuple[1])

        # Strip out indices used for sorting.
        values    = []
        for value in raw_values:
            values.append(value[0])

        return values

    # Returns a fourier descriptor that is invariant to rotation and boundary starting point.
    def make_rotation_sp_invariant(self, fourier_desc):
        for index, value in enumerate(fourier_desc):
          fourier_desc[index] = np.absolute(value)

        return fourier_desc

    # Returns a fourier descriptor that is invariant to scale.
    def make_scale_invariant(self, fourier_desc):
        first_val    = fourier_desc[0]

        for index, value in enumerate(fourier_desc):
            fourier_desc[index]    = value / first_val

        return fourier_desc

    # Returns a fourier descriptor that is invariant to translation.
    def make_translation_invariant(self, fourier_desc):
        return fourier_desc[1:len(fourier_desc)]

