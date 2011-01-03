import cv
import os
import math
import random
import numpy

def rgba(r,g,b,a):
    v = cv.RGB(r,g,b)
    return v[:3] + (a,)
    
def is_grayscale(im):
    if im.nChannels == 1:
        return True
    h,s,v = split(rgb2hsv(im))
    if cv.CountNonZero(h) == 0 == cv.CountNonZero(s):
        return True
    return False

def new_from(im, depth=None, nChannels=None, size=None):
    if depth is None:
        depth = im.depth
    if nChannels is None:
        nChannels = im.nChannels
    if size is None:
        size = cv.GetSize(im)
    
    return cv.CreateImage(size, depth, nChannels)

def crop(im, rect):
    cropped = new_from(im, size=rect[2:])
    cv.SetImageROI(im, tuple(rect))
    cv.Copy(im, cropped)
    cv.ResetImageROI(im)
    return cropped
    
def random_cropped_region(im, size):
    isize = cv.GetSize(im)
    s = (min(size[0], isize[0]), min(size[1], isize[1]))
    if s == isize:
        return im
    mx, my = isize[0] - s[0], isize[1] - s[1]
    x, y = random.randrange(mx), random.randrange(my)
    print x,y, isize, size
    print (x,y,x+size[0],y+size[1])
    return crop(im, (x,y,size[0],size[1]))
    
def op(cvFunc, im1, im2, *images):
    new_im = new_from(im1)
    cvFunc(im1, im2, new_im)
    for im in images:
        cvFunc(new_im, im, new_im)
    return new_im
    
def add(im1, im2, *images): return op(cv.Add, im1, im2, *images)
def sub(im1, im2, *images):return op(cv.Sub, im1, im2, *images)
def multiply(im1, im2, *images): return op(cv.Mul, im1, im2, *images)
def And(im1, im2, *images): return op(cv.And, im1, im2, *images)
def Or(im1, im2, *images): return op(cv.Or, im1, im2, *images)
def Xor(im1, im2, *images): return op(cv.Xor, im1, im2, *images)
def absDiff(im1, im2, *images):return op(cv.AbsDiff, im1, im2, *images)

def blend(im1, im2, alpha=0.5):
    new_im = new_from(im1)
    cv.AddWeighted(im1, alpha, im2, 1-alpha, 0.0, new_im)
    return new_im
    
def gaussian(im, size=9):
    if type(size) in (int, float, long):
        size = (size, size)
    new_im = new_from(im)
    cv.Smooth(im, new_im, cv.CV_GAUSSIAN, size[0], size[1])
    return new_im

def erode(im, element=None, iterations=1):
    new_im = new_from(im)
    cv.Erode(im, new_im, element, iterations)
    return new_im
    
def dilate(im, element=None, iterations=1):
    new_im = new_from(im)
    cv.Dilate(im, new_im, element, iterations)
    return new_im

def resize(im, size=None, by_percent=None, method=cv.CV_INTER_LINEAR):
    assert size != None or by_percent != None
    if size is not None:
        im_size = cv.GetSize(im)
        size = list(size)
        if size[0] is None:
            size[0] = size[1] / float(im_size[1]) * im_size[0]
        if size[1] is None:
            size[1] = size[0] / float(im_size[0]) * im_size[1]
    
    if by_percent is not None:
        size = list(cv.GetSize(im))
        if type(by_percent) in (list, tuple):
            size[0] *= by_percent[0]
            size[1] *= by_percent[1]
        else:
            size[0] *= by_percent
            size[1] *= by_percent
    size = (int(size[0]), int(size[1]))
    resized_im = new_from(im, size=size)
    cv.Resize(im, resized_im, method)
    return resized_im

def max_size(im, size, method=cv.CV_INTER_LINEAR):
    im_size = cv.GetSize(im)
    if im_size[0] > size[0]:
        return resize(im, (size[0],None), method=method)
    if im_size[1] > size[1]:
        return resize(im, (None, size[1]), method=method)
    return im

def load(path, grayscale=False, max_size=None):
    path = os.path.join('data', path)
    if not os.path.exists(path):
        raise TypeError, "File does not exist: "+repr(os.path.abspath(path))

    if grayscale:
        return cv.LoadImage(path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    return cv.LoadImage(path, cv.CV_LOAD_IMAGE_COLOR)

def invert(im):
    inv = new_from(im)
    cv.SubRS(im, cv.ScalarAll(int('1' * im.depth, 2)), inv)
    return inv

def threshold(im, threshold=128, type=cv.CV_THRESH_BINARY, max_value=255):
    if not is_grayscale(im):
        im = rgb2gray(im)
    new_im = new_from(im)
    cv.Threshold(im, new_im, threshold, max_value, type)
    return new_im
    
def dft(im, flags,):
    new_im = new_from(im)
    
def split(im):
    if im.nChannels == 3:
        imr = new_from(im, nChannels=1)
        img = new_from(im, nChannels=1)
        imb = new_from(im, nChannels=1)
        cv.Split(im, imr, img, imb, None)
        return (imr, img, imb)
    else:
        return (im,)

def merge(im1, im2, im3, im4=None):
    assert 1 == im1.nChannels == im2.nChannels == im3.nChannels
    new_im = new_from(im1, nChannels=3)
    cv.Merge(im1, im2, im3, im4, new_im)
    return new_im
    
def equalize_hist(im):
    new_im = new_from(im)
    cv.EqualizeHist(im, new_im)
    return new_im

def rgb2hsv(im):
    hsv = new_from(im)
    cv.CvtColor(im, hsv, cv.CV_BGR2HSV)
    return hsv

def hsv2rgb(im):
    rgb = new_from(im)
    cv.CvtColor(im, rgb, cv.CV_HSV2BGR)
    return rgb

def rgb2luv(im):
    luv = new_from(im)
    cv.CvtColor(im, luv, cv.CV_BGR2Luv)
    return luv

def luv2rgb(im):
    rgb = new_from(im)
    cv.CvtColor(im, rgb, cv.CV_Luv2BGR)
    return rgb
    
def rgb2gray(im):
    if im.nChannels == 1:
        return im
    gray = new_from(im, cv.IPL_DEPTH_8U, nChannels=1)
    cv.CvtColor(im, gray, cv.CV_RGB2GRAY)
    return gray
    
def gray2rgb(im):
    if im.nChannels==3:
        return im
    color = new_from(im, cv.IPL_DEPTH_8U, nChannels=3)
    cv.CvtColor(im, color, cv.CV_GRAY2RGB)
    return color

def laplace(im):
    im = rgb2gray(im)
    new_im = new_from(im, depth=cv.IPL_DEPTH_16S)
    cv.Laplace(im, new_im)
    cv.ConvertScaleAbs(new_im, im)
    return im
    
def edges(im, threshold1=50, threshold2=150, aperture_size=3):
    edges = new_from(im, cv.IPL_DEPTH_8U, 1)
    gray = rgb2gray(im)
    cv.Canny(gray, edges, threshold1, threshold2, aperture_size)
    return edges

def auto_edges(im, starting_threshold=50, percentage=0.2):
    size = cv.GetSize(im)
    total = size[0] * size[1] * percentage
    e = edges(im, starting_threshold, starting_threshold*3)
    while cv.CountNonZero(e) > total:
        starting_threshold += 10
        e = edges(im, starting_threshold, starting_threshold*3)
    return e

def corners(im, max_corners=100, quality=0.1, min_dist=5, block_size=3, use_harris=False,
    mask=None, k=0.04):
    eig = new_from(im, depth=cv.IPL_DEPTH_32F, nChannels=1)
    tmp = new_from(im, depth=cv.IPL_DEPTH_32F, nChannels=1)
    gray = rgb2gray(im)
    corners = cv.GoodFeaturesToTrack(gray, eig, tmp, max_corners, quality, min_dist,
        mask, block_size, use_harris, k)
    #cv.Scale(eig, eig, 100, 0.00)
    return corners

size = [10, 20]

def show(im, title="Image"):
    global size # I know, bad practice...
    cv.NamedWindow(title)
    im2 = max_size(im, (640, 480))
    cv.ShowImage(title, im2)
    cv.MoveWindow(title, size[0], size[1])
    size[0] += 20
    size[1] += 20
    
def sobel(im, xorder=1, yorder=1):
    recommended_size = {
        1: 3, 2: 3, 3: 5,
        4: 5, 5:7, 6: 7, 7:9, 8:9
    }
    im = rgb2gray(im)
    new_im = cv.CreateImage(cv.GetSize(im), cv.IPL_DEPTH_16S, 1)
    cv.Sobel(im, new_im, xorder, yorder,
        apertureSize=recommended_size[max(xorder, yorder)])
    cv.ConvertScaleAbs(new_im, im)
    return im
    
def save(im, path):
    #root = os.path.join('output')
    root = os.path.dirname(path)
    try:
        os.makedirs(root)
    except OSError:
        pass
    path = os.path.join(root, path)
    cv.SaveImage(path, im)

def set_pixel(im, x, y, rgb):
    if type(rgb) in (tuple, list) and len(rgb) == 3:
        rgb = (rgb[2], rgb[1], rgb[0])
    cv.Set2D(im, y, x, rgb)

def neighbor_map(im, func, nbr_size=3):
    oim = new_from(im)
    size = cv.GetSize(oim)
    nbr_size = int(nbr_size / 2)
    size = tuple((x - nbr_size for x in size))
    for x in xrange(nbr_size, size[0]):
        for y in xrange(nbr_size, size[1]):
            value = func(im[y-nbr_size:y+nbr_size+1, x-nbr_size:x+nbr_size+1])
            set_pixel(oim, x, y, value)
    return oim

def draw_points(im, points, radius=2, color=cv.RGB(255,0,0), thickness=1):
    for pt in points:
        cv.Circle(im, tuple(map(int, pt)), radius, color, thickness)
    return im
    
    
def cv2array(im):
    depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

    arrdtype=im.depth
    a = numpy.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
    a.shape = (im.height,im.width,im.nChannels)
    return a


def array2cv(a):
    dtype2depth = {
      'uint8':   cv.IPL_DEPTH_8U,
      'int8':    cv.IPL_DEPTH_8S,
      'uint16':  cv.IPL_DEPTH_16U,
      'int16':   cv.IPL_DEPTH_16S,
      'int32':   cv.IPL_DEPTH_32S,
      'float32': cv.IPL_DEPTH_32F,
      'float64': cv.IPL_DEPTH_64F,
    }
    try:
        nChannels = a.shape[2]
    except:
        nChannels = 1
        cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
            dtype2depth[str(a.dtype)],
            nChannels)
        cv.SetData(cv_im, a.tostring(),
               a.dtype.itemsize*nChannels*a.shape[1])
    return cv_im