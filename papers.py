import cv
import image
import random

def blurry_histogram(im, num_samples=300, offset=1):
    im = image.rgb2gray(im)
    size = cv.GetSize(im)
    used = set([])
    i = 0
    diffs = {}
    while i < num_samples:
        # we can't use the first row of pixels
        x = random.randrange(0, size[0])
        y = random.randrange(offset, size[1])
        if (x,y) not in used:
            pixel1 = cv.Get2D(im, y, x)
            pixel2 = cv.Get2D(im, y-offset, x)
            diff = tuple(map(lambda a,b: a-b, pixel1, pixel2))
            if diff not in diffs:
                diffs[diff] = 0
            diffs[diff] += 1
            used = used.union([(x,y)])
            i += 1
    max_i = max_v = 0
    second_max_i = second_max_v = 0
    for key,val in diffs.iteritems():
        if max_v < val:
            second_max_i, second_max_v = max_i, max_v
            max_v, max_i = val, key
    return (max_v - second_max_v) / abs(max_i[0] - second_max_i[0])