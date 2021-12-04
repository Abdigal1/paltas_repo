from .Phantom_stats_trans.phantom_stats_methods import entropy_mark

class entropy_mark_transform(object):
    """Crop randomly the image in a sample.

    Args:
        regular (bool): regular o non regular segmentation mask
    output:

    """

    def __init__(self, ImType='PhantomRGB'):
        if ImType in ['PhantomRGB','SenteraRGB']:
            self.ImType=ImType
        else:
            print("only RGB types")

    def __call__(self, sample):
        print(self.ImType)
        image, landmarks = sample[self.ImType], sample['landmarks']

        image=entropy_mark(image)

        return {'image': image, 'landmarks': landmarks}