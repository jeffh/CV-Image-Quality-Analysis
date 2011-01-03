import image
import cv
from windows import GrayRGBWindow
from grid import Grid

from histogram import GrayscaleHist

def average(seq):
    if len(seq) == 0:
        return 0
    s = sum(seq)
    return s / float(len(seq))

def boolean(score, threshold=32): # restricted to the life, universe, and everything
    (mean, stddev) = score
    return (mean < threshold) and stddev < 64
    #return mean - stddev/2 <= 0 or mean < 32

def score_hist(hist):
    #a1, a2 = average(hist.count(slice(None, 32))), average(hist.count(slice(32, None)))
    #return min(a1, a2) / max(a1, a2)
    return hist.mean(), hist.stddev()

requires_result_from = []
def measure(im, debug=False):
    gray = image.rgb2gray(im)
    _,s,v = image.split(image.rgb2hsv(im))
    h = GrayscaleHist(bins=64).use_image(v)
    s = GrayscaleHist(bins=64).use_image(s)
    scores = [score_hist(h)]
    
    if debug:
        image.show(v, "1. Value")
        image.show(h.to_img(), "2. Value Histogram")
        image.show(s.to_img(), "2. Saturation Histogram")
        print score_hist(s)
    
    return (
        None, #h.to_img(),
        scores[0],
    )