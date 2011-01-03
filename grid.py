import cv
import image

class Grid(object):
    def __init__(self, size):
        super(Grid, self).__init__()
        self.size = size
        
    def split_into(self, im, count):
        if type(count) not in (list, tuple):
            count = (count, count)
        box_w, box_h = self.size[0]/count[0], self.size[1]/count[1]
        images = []
        for x in range(count[0]):
            for y in range(count[1]):
                box = (box_w * x, box_h * y, box_w-1, box_h-1)
                images.append(image.crop(im, box))
        return images
        
    def draw_regions(self, im, color=cv.RGB(255,255,255), thickness=cv.CV_FILLED):
        t1, t2 = (self.size[0]/3, self.size[1]/3)
        b1, b2 = t1/3, t2/3
        
        cv.Rectangle(im, (t1-b1, t2-b2), (t1+b1, t2+b2), color, thickness)
        cv.Rectangle(im, (t1*2-b1, t2-b2), (t1*2+b1, t2+b2), color, thickness)
        cv.Rectangle(im, (t1-b1, t2*2-b2), (t1+b1, t2*2+b2), color, thickness)
        cv.Rectangle(im, (t1*2-b1, t2*2-b2), (t1*2+b1, t2*2+b2), color, thickness)
        return im
        
    def draw_lines(self, im, color=cv.RGB(255,255,255), thickness=5):
        t1, t2 = (self.size[0]/3, self.size[1]/3)
        b1, b2 = t1/3, t2/3
        cv.Line(im, (t1, 0), (t1, self.size[1]), color, thickness=thickness)
        cv.Line(im, (t1*2, 0), (t1*2, self.size[1]), color, thickness=thickness)
        cv.Line(im, (0, t2), (self.size[0], t2), color, thickness=thickness)
        cv.Line(im, (0, t2*2), (self.size[0], t2*2), color, thickness=thickness)
        
        return im

    def draw(self, im, color=cv.RGB(255,0,0), thickness=10):
        self.draw_lines(im, color, abs(thickness))
        self.draw_regions(im, color, thickness)
        
        return im
        
    def split_in_four(self, im):
        half = self.size[0]/2, self.size[1]/2
        top_left = image.crop(im, (0, 0, half[0], half[1]))
        top_right = image.crop(im, (half[0], 0, half[0], half[1]))
        bottom_left = image.crop(im, (0, half[1], half[0], half[1]))
        bottom_right = image.crop(im, (half[0], half[1], half[0], half[1]))
        return top_left, top_right, bottom_left, bottom_right