import numpy

from radiometric_normalization import \
    time_stack, pif, transformation, validation, gimage


def generate_luts(candidate_path, reference_paths, config=None):
    if config is None:
        config = {'time_stack_method': 'identity',
                  'pif_method': 'identity',
                  'transformation_method': 'linear_relationship'}

    time_stack_image = time_stack.generate(reference_paths,
                                           method=config['time_stack_method'])
    pifs = pif.generate(candidate_path,
                        time_stack_path=time_stack_image,
                        method=config['pif_method'])
    transformations = transformation.generate(
        pifs, method=config['transformation_method'])
    luts = transformation.transformations_to_luts(transformations)
    return luts


def apply_luts(input_path, luts, output_path):
    def apply_lut(band, lut):
        'Changes band intensity values based on intensity look up table (lut)'
        if lut.dtype != band.dtype:
            raise Exception(
                "Band ({}) and lut ({}) must be the same data type.").format(
                band.dtype, lut.dtype)
        return numpy.take(lut, band, mode='clip')

    img = gimage.load(input_path)
    for i in range(len(img.bands)):
        img.bands[i] = apply_lut(img.bands[i], luts[i])
    img.save(output_path)


def validate_luts(input_path, reference_path, config=None):
    if config is None:
        config = {'validation_method': 'identity'}
    score = validation.get_score(input_path,
                                 reference_path,
                                 method=config['validation_method'])
    return score
