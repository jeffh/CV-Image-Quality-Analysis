import image
import cv
import numpy

class GrayscaleHist(object):
    def __init__(self, bins=32, value_range=(0,255), scale=10):
        self.bins = bins
        self.value_range = value_range
        self.scale = scale
        
        self.kind = 'opencv'
    
    def use_image(self, im):
        self.kind = 'opencv'
        self.hist = cv.CreateHist([self.bins], cv.CV_HIST_ARRAY, [self.value_range], 1)
        cv.CalcHist([im], self.hist)
        self.min_value, self.max_value, _, _ = cv.GetMinMaxHistValue(self.hist)
        return self
        
    def use_array(self, arr):
        self.kind = 'numpy'
        if not isinstance(arr, numpy.ndarray):
            arr = numpy.array(list(arr))
        self.hist, _ = numpy.histogram(arr, self.bins, self.value_range)
        self.min_value, self.max_value = min(self.hist), max(self.hist)
        return self
        
    def use_array_as_hist(self, arr):
        self.kind = 'numpy'
        if not isinstance(arr, numpy.ndarray):
            arr = numpy.array(list(arr))
        self.hist = arr
        self.bin = len(self.hist)
        self.min_value, self.max_value = min(self.hist), max(self.hist)
        return self
        
    def mean(self):
        if self.kind == 'numpy':
            return self.hist.mean()
        return numpy.array(list(self)).mean()
    
    def stddev(self):
        if self.kind == 'numpy':
            return self.hist.std()
        return numpy.array(list(self)).std()
        
    def count(self, value):
        if isinstance(value, slice):
            return [self.count(i) for i in range(*value.indices(len(self)))]
        if self.kind == 'numpy':
            return self.hist[value]
        return cv.QueryHistValue_1D(self.hist, value)
         
    def intensity_at_bin(self, value):
        if isinstance(value, slice):
            return [self.intensity_at_bin(i) for i in range(*value.indices(len(self)))]
        
        if self.value_range:
            max_value = float(self.value_range[1])
        else:
            max_value = float(self.max_value)
    
        return round(self.count(value) * max_value / float(max(self.max_value, 0.0001)))
        
    def index(self, intensity):
        l = list(self)
        return l.index(intensity)
        
    def __getitem__(self, bin_num):
        if isinstance(bin_num, slice):
            return [self[i] for i in range(*bin_num.indices(len(self)))]
        return self.intensity_at_bin(bin_num)
        
    def __len__(self):
        return self.bins
        
    def __iter__(self):
        for bin_num in range(self.bins):
            yield self.intensity_at_bin(bin_num)
    
    def to_array(self):
        return numpy.array(list(self))
        
    def draw(self, im, color=cv.RGB(0,0,0), offset=(0,0)):
        height = cv.GetSize(im)[1]
        
        pad = 2
        
        if self.value_range:
            max_value = float(self.value_range[1])
        else:
            max_value = float(self.max_value)
        
        for b in range(self.bins):
            value = self.intensity_at_bin(b)

            cv.Rectangle(im,
                (offset[0] + b*self.scale + pad,
                 offset[1] + height-int(value/max_value * height)),
                (offset[0] + int((b+1)*self.scale - pad),
                 offset[1] + height),
                color, cv.CV_FILLED)
        return im
        
    def to_img(self, height=200, color=cv.RGB(0,0,0), offset=(0,0), bg=(255,255,255)):
        im = cv.CreateImage((self.bins * self.scale, height), 8, 3)
        cv.Set(im, bg)
        return self.draw(im, color, offset=(0,0))