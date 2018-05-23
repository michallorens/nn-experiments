from PIL import Image


class Scale:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)


class Slice:
    def __init__(self, num_slices, slice):
        self.num_slices = num_slices
        self.slice = slice

    def __call__(self, img):
        width, height = img.size
        upper = int((self.slice - 1) * height / self.num_slices)
        lower = int(self.slice * height / self.num_slices)
        left = 0

        box = (left, upper, width, lower)
        return img.crop(box)
