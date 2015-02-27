import unittest
import numpy

import pif


class Tests(unittest.TestCase):

    def test_create_pif_dict(self):
        nodata = 2 ** 15 - 1

        # Three bands of 2 by 2 pixels
        time_stack = numpy.array([numpy.array([[nodata, 1],
                                               [nodata, 2]],
                                              dtype='uint16'),
                                  numpy.array([[nodata, 7],
                                               [8, 6]],
                                              dtype='uint16'),
                                  numpy.array([[2, 9],
                                               [nodata, 1]],
                                              dtype='uint16')])
        weighting = numpy.array([[0, 65535],
                                 [0, 65535]],
                                dtype='uint16')

        candidate = numpy.array([numpy.array([[8, 6],
                                              [nodata, 7]],
                                             dtype='uint16'),
                                 numpy.array([[9, 1],
                                              [nodata, 4]],
                                             dtype='uint16'),
                                 numpy.array([[3, 3],
                                              [nodata, 1]],
                                             dtype='uint16')])
        candidate_mask = numpy.array([[65535, 65535],
                                      [0, 65535]],
                                     dtype='uint16')

        golden_output_list = [{'coordinates': (0, 1),
                               'weighting': 65535,
                               'reference': numpy.array([1, 7, 9]),
                               'candidate': numpy.array([6, 1, 3])},
                              {'coordinates': (1, 1),
                               'weighting': 65535,
                               'reference': numpy.array([2, 6, 1]),
                               'candidate': numpy.array([7, 4, 1])}]

        output_list = pif._filter_nodata(time_stack, weighting,
                                         candidate, candidate_mask)

        for pixel in range(len(output_list)):
            self.assertEqual(output_list[pixel]['coordinates'],
                             golden_output_list[pixel]['coordinates'])
            self.assertEqual(output_list[pixel]['weighting'],
                             golden_output_list[pixel]['weighting'])
            self.assertEqual(output_list[pixel]['reference'].all(),
                             golden_output_list[pixel]['reference'].all())
            self.assertEqual(output_list[pixel]['candidate'].all(),
                             golden_output_list[pixel]['candidate'].all())


if __name__ == '__main__':
    unittest.main()
