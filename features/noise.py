import image
import cv
from numpy import array
from grid import Grid

class Noise(object):
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev
    
    def __call__(self, im):
        size = cv.GetSize(im)
        #print size
        pixel = im[int(size[0]/2), int(size[1]/2)]
        mean, stddev = [x[0] for x in cv.AvgSdv(im)]
        #if (value < self.mean-self.stddev or value >= self.mean+self.stddev):
        #if stddev > self.stddev:
        if stddev > self.stddev:
            return 255
        return 0

def count_neighbors(im):
    size = cv.GetSize(im)
    mx, my = int(size[0]/2), int(size[1]/2)
    pixel = im[mx, my]
    if pixel == 0:
        return 0
    c = cv.CountNonZero(im)
    return c
    
def noise_per_grid(images):
    size = cv.GetSize(images[0])
    total = float(size[0] * size[1])
    scores = []
    for im in images:
        scores.append(cv.CountNonZero(im) / total)
    return scores
    
def boolean(score, threshold=0.13, diff_threshold=0.05):
    diff = score[1]-score[0]
    diff_mean, diff_std = score[2], score[3]
    uv_score, avg_std = score[4]
    
    #return uv_score > 0.90 or uv_score < 0.1
    
    #return (diff_std > 25)
    if (score[0] > threshold and score[1] > threshold and \
        (abs(diff) > diff_threshold or diff < 0)):
        return uv_score > 0.9 and avg_std > 60
    return False

requires_result_from = []
"""
Too "clever", too slow

def measure(im, debug=False):
    #im = image.random_cropped_region(im, (640, 480))
    #l = image.laplace(im)
    l = image.sobel(im, xorder=2, yorder=2)
    #l = image.dilate(l)
    size = cv.GetSize(l)
    mean, stddev = map(lambda x: x[0], cv.AvgSdv(l))
    n = Noise(mean, stddev)
    l = image.neighbor_map(l, n, nbr_size=3)
    edges = image.threshold(image.auto_edges(im, percentage=0.1), threshold=1)
    cv.Set(l, 0, edges)

    l = image.neighbor_map(l, count_neighbors, nbr_size=3)
    score = cv.Sum(l)[0] / float((size[0]-2) * (size[1]-2) * 9)
    
    if debug:
        image.show(edges, "Edges")
        image.show(image.threshold(l, threshold=1), "Final")
        image.show(image.sub(im, image.gray2rgb(l)), "Final Sub")
    
    return image.threshold(l, threshold=1), score
"""

    
def measure(im, debug=False):
    gray = image.rgb2gray(im)
    size = cv.GetSize(im)
    total = float(size[0] * size[1])
    l = image.sub(gray, image.gaussian(gray, 5))
    l2 = image.sub(gray, image.gaussian(gray, 9))
    edges = image.dilate(image.auto_edges(im, percentage=0.2))
    if debug:
        image.show(image.threshold(l, threshold=1), "Before Edge Removal (kernel=5)")
        image.show(image.threshold(l2, threshold=1), "Before Edge Removal (kernel=9)")
    cv.Set(l, 0, image.threshold(edges, threshold=1))
    cv.Set(l2, 0, image.threshold(edges, threshold=1))
    
    l = image.threshold(l, threshold=1)
    l2 = image.threshold(l2, threshold=1)
    
    
    
    if debug:
        image.show(image.threshold(edges, threshold=1), "Edges")
        image.show(l, "After Edge Removal (kernel=5)")
        image.show(l2, "After Edge Removal (kernel=9)")
        
    noise2 = image.new_from(gray)
    cv.EqualizeHist(gray, noise2)
    cv.AbsDiff(noise2, gray, noise2)
    cv.Set(noise2, 0, image.threshold(image.sobel(im, xorder=2, yorder=2), threshold=4))
    diff = image.cv2array(noise2)
    if debug:
        image.show(noise2, "DIFF")
        print "M", diff.mean(), "S", diff.std()
    diff_stat = (diff.mean(), diff.std())
    percent_noise = cv.CountNonZero(noise2) / total
    if debug:
        image.show(noise2, "NOISE2")
        


    # magical, I don't understand how this works
    _, sat, _ = image.split(image.rgb2hsv(im))
    edges = image.auto_edges(im)
    l,u,v = tuple(map(image.equalize_hist, image.split(image.rgb2luv(im))))
    u,v = tuple(map(image.gaussian, (u,v)))
    if debug:
        image.show(l, "1. L")
        image.show(u, "1. U")
        image.show(v, "1. V")
    la,ua,va,uva = tuple(map(image.cv2array, (l,u,v, image.And(l,u,v))))
    test = image.new_from(gray)
    test2 = image.new_from(gray)
    cv.Xor(u,v,test)
    if debug:
        image.show(test, "2. U Xor V")
    cv.Set(test, 0, image.dilate(edges))
    #cv.Set(test, 0, image.invert(image.threshold(sat, threshold=8)))
    uv_score = cv.CountNonZero(test) / total
    if debug:
        image.show(test, "3. U Xor V - dilate(Edges) - invert(threshold(Saturation))")

    g = Grid(size)
    images = map(image.cv2array, g.split_into(test, 6))
    arr = image.cv2array(test)
    avg_mean, avg_std = arr.mean(), arr.std()


    #ms = [(a.mean(), a.std()) for a in images]
    #min_mean = min_std = 255
    #max_mean = max_std = 0
    #for m,s in ms:
    #    min_mean = min(min_mean, m)
    #    min_std = min(min_std, s)
    #    max_mean = max(max_mean, m)
    #    max_std = max(max_std, s)
    #if debug:
    #    print min_mean, min_std
    #    print avg_mean, avg_std
    #    print max_mean, max_std
    #
    #score = uv_score, min_mean, avg_mean, avg_std, max_mean
    uv_score = uv_score, avg_std

    score = cv.CountNonZero(l) / total,  cv.CountNonZero(l2) / total, \
        diff_stat[0], diff_stat[1], uv_score
    
    return l, score
    

def boolean((uv_score, avg_std)):
    return (uv_score > 0.9 or uv_score < 0.1) and avg_std >= 60


def measure(im, debug=False):
    
    gray = image.rgb2gray(im)
    size = cv.GetSize(im)
    total = float(size[0] * size[1])
    edges = image.auto_edges(im)

    _, sat, val = image.split(image.rgb2hsv(im))
    edges = image.auto_edges(im)
    l,u,v = tuple(map(image.equalize_hist, image.split(image.rgb2luv(im))))
    u,v = tuple(map(image.gaussian, (u,v)))
    if debug:
        image.show(l, "1. L")
        image.show(u, "1. U")
        image.show(v, "1. V")
    la,ua,va,uva = tuple(map(image.cv2array, (l,u,v, image.And(l,u,v))))
    test = image.new_from(gray)
    test2 = image.new_from(gray)
    cv.Xor(u,v,test)
    #cv.AbsDiff(u,v, test2)
    if debug:
        #cv.Threshold(test, test, 32, 255, cv.CV_THRESH_BINARY)
        image.show(test, "2. U Xor V")
        #image.show(test2, "TEST 2")
        #test = image.dilate(test)
    cv.Set(test, 0, image.dilate(edges))
    #cv.Set(test, 0, image.invert(image.threshold(sat, threshold=8)))
    uv_score = cv.CountNonZero(test) / total
    if debug:
        image.show(test, "3. U Xor V - dilate(Edges) - invert(threshold(Saturation))")
    
    arr = image.cv2array(test)
    avg_mean, avg_std = arr.mean(), arr.std()
    
    score = uv_score, avg_std
    
    return test, score

### NEW METHOD

def boolean((mean, std, over)):
    # over <= 3
    n = 16 # grid size
    return over <= 0.05*(n*n) and mean > 70
    
def measure(im, debug=False):       
    gray = image.rgb2gray(im)
    size = cv.GetSize(im)
    total = float(size[0] * size[1])
    edges = image.auto_edges(im)

    hue, sat, val = tuple(map(image.equalize_hist, image.split(image.rgb2hsv(im)) ))
    l,u,v = tuple(map(image.equalize_hist, image.split(image.rgb2luv(im))))
    
    values = []
    if debug:
        image.show(l, "L")
        image.show(val, "Value")
    sat = image.threshold(val,255-32)#image.And(val, sat)
    if debug:
        image.show(sat, "Thresh")
    #cv.And(val, l, val)
    cv.Sub(l, sat, l)
    cv.Set(l, 0, image.dilate(edges, iterations=3))
    if debug:
        image.show(l, "L - Value")
    val = l
    g = Grid(cv.GetSize(val))
    images = g.split_into(val, 16)
    arr = image.cv2array(val)
    avgmean, avgstd = arr.mean(), arr.std()
    for i in images:
        a = image.cv2array(i)
        mean, std = abs(a.mean() - avgmean), max(a.std(), 0)
        values.append((mean+std))
    
    if debug:
        print values
        print "AVG", avgmean, avgstd
        image.show(val, "Result")
        
    return val, (avgmean, avgstd, len([v for v in values if v > avgstd*2]))