import image
import cv
import os

DIR = os.path.join(*('data/haarcascades'.split('/')))
face_cascades = [os.path.join(DIR,x) for x in (
'haarcascade_frontalface_alt.xml',
'haarcascade_frontalface_alt2.xml',
'haarcascade_frontalface_alt_tree.xml',
'haarcascade_frontalface_default.xml',
'haarcascade_profileface.xml',
)]

eye_cascades = [os.path.join(DIR,x) for x in (
'haarcascade_eye.xml',
'haarcascade_eye_tree_eyeglasses.xml',
'haarcascade_mcs_eyepair_big.xml',
'haarcascade_mcs_eyepair_small.xml',
'haarcascade_mcs_lefteye.xml',
'haarcascade_lefteye_2splits.xml',
'haarcascade_mcs_righteye.xml',
'haarcascade_righteye_2splits.xml',
)]

mouth_cascades = [os.path.join(DIR,x) for x in (
'haarcascade_mcs_mouth.xml',
)]

body_cascades = [os.path.join(DIR,x) for x in (
'haarcascade_fullbody.xml',
'haarcascade_upperbody.xml',
'haarcascade_lowerbody.xml',
'haarcascade_mcs_upperbody.xml',
)]

def detect_skin(im, debug=False):
    hsv = image.rgb2hsv(im)
    
    if debug:
        image.show(hsv, 'hsv')
    h,s,v = image.split(hsv)
    
    if cv.CountNonZero(h) == cv.CountNonZero(s) == 0:
        white = image.new_from(im)
        cv.Set(white, 255)
        return white
    
    if debug:
        image.show(h, "Hue")
        image.show(s,"sat1")
    
    h_rng = 0, 46
    s_rng = 48, 178
    
    h = image.threshold(image.gaussian(h, 5), threshold=h_rng[1], type=cv.CV_THRESH_TOZERO_INV)
    h = image.threshold(h, threshold=h_rng[0], type=cv.CV_THRESH_TOZERO)
    h = image.threshold(h, threshold=1)
    
    s = image.threshold(image.gaussian(s, 5), threshold=s_rng[1], type=cv.CV_THRESH_TOZERO_INV)
    s = image.threshold(s, threshold=s_rng[0], type=cv.CV_THRESH_TOZERO)
    if debug:
        image.show(s,"sat2")
    s = image.threshold(s, threshold=1)
    
    v = image.dilate(image.erode(image.And(s, h)))
    
    
    #im = image.hsv2rgb(image.merge(h,s,v))
    if debug:
        image.show(v, "Human")
    return image.threshold(v, threshold=1)

im_scale = 0.5    

def detect(im, cascade, haar_scale=1.2, min_neighbors=2, min_size=(20,20)):
    haar_flags = 0

    gray = image.rgb2gray(im)
    small = image.resize(gray, by_percent=im_scale)
    cv.EqualizeHist(small, small)
    objs = cv.HaarDetectObjects(small, cascade, cv.CreateMemStorage(0),
        haar_scale, min_neighbors, cv.CV_HAAR_DO_CANNY_PRUNING, min_size)
    return [Rect(r,n,cv.GetSize(im)) for r,n in objs]

def draw_objects(im, objs, color=cv.RGB(255,0,0), thickness=1):
    for ((x, y, w, h), n) in objs:
        #print n
        if n < 4: continue
        pt1 = tuple(map(int, (x / im_scale, y / im_scale)))
        pt2 = tuple(map(int,((x+w) / im_scale, (y+h) / im_scale)))
        cv.Rectangle(im, pt1, pt2, color, thickness=thickness)
    return im
    
class Rect(object):
    def __init__(self, (x, y, w, h), n=None, im_size=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.n = n
        self.im_size = im_size
        
    def __contains__(self, rect):
        x1, x2 = self.x, self.x+self.w
        y1, y2 = self.y, self.y+self.h
        return x1 <= rect.x <= x2 and y1 <= rect.y <= y2 and \
            x1 <= rect.x + rect.w <= x2 and y1 <= rect.y + rect.h <= y2
            
    def draw(self, im, color=(255,255,255), thickness=4):
        im_size = cv.GetSize(im)
        im2_size = self.im_size
        size_ratio = (
            (im_size[0] / float(im2_size[0])) / im_scale,
            (im_size[1] / float(im2_size[1])) / im_scale,
        )
            
        pt1 = int(self.x * size_ratio[0]), int(self.y * size_ratio[1])
        pt2 = int((self.x+self.w) * size_ratio[0]), int((self.y+self.h) * size_ratio[1])
        cv.Rectangle(im, pt1, pt2, color, thickness=thickness)
        return im
    
    def intersects_with(self, rect):
        return self.x <= rect.x <= self.x+self.w or self.y <= rect.y <= self.y+self.h or \
            rect.x <= self.x <= rect.x+rect.w or rect.y <= self.y <= rect.y+rect.h
    
    def area(self):
        return self.w * self.h
        
    def __cmp__(self, rect):
        return cmp(self.area(), rect.area())
        
    def __eq__(self, rect):
        return (self.x, self.y, self.w, self.h, self.n) == \
            (rect.x, rect.y, rect.w, rect.h, rect.n)
    
    def __hash__(self):
        return hash((self.x, self.y, self.w, self.h))
        
    def __repr__(self):
        return "Rect((%d, %d, %d, %d), %d)" % (
            self.x, self.y, self.w, self.h, self.n,
        )
    
    def intersected_area(self, rect):
        if not self.intersects_with(rect):
            return 0
        width = min(self.x + self.w, rect.x + rect.w) - max(self.x, rect.x)
        height = min(self.y + self.h, rect.y + rect.h) - max(self.y, rect.y)
        return width * height
    
    def to_tuple(self):
        return ((self.x, self.y, self.w, self.h), self.n)
    
    def __iter__(self):
        return iter(((self.x, self.y, self.w, self.h), self.n))
    
def filter_overlap(rects, overlap_ratio=0.75):
    "Filters overlapping objects as one. (Picks a random overlapped object as it)."
    to_remove = []
    for i,r1 in enumerate(rects):
        for j,r2 in enumerate(rects):
            if i <= j: continue
            a = min(r1.area(), r2.area())
            if r1.intersected_area(r2) > a * overlap_ratio:
                to_remove.append(max(r1, r2))
    to_remove = set(to_remove)
    filtered_rects = [r for r in rects if r not in to_remove]
    counts = {}
    for r in filtered_rects:
        for r2 in rects:
            if r == r2: continue
            a = min(r1.area(), r2.area())
            if r1.intersected_area(r2) > a * overlap_ratio:
                counts[r] = counts.get(r, 0) + 1
    return counts

def boolean(faces):
    return faces < 1

#cascade = cv.Load(face_cascades[-2])

requires_result_from = []
def measure(im, debug=False):
    im2 = image.max_size(im, (800, 600))
    
    b,g,r = image.split(im2)
    #cv.EqualizeHist(r,r)
    ##cv.EqualizeHist(g,g)
    ##cv.EqualizeHist(b,b)
    im2 = image.merge(b,g,r)
    
    eyes = 0
    #objs = []
    #for cascade in eye_cascades:
    #    print cascade
    #    cascade = cv.Load(cascade)
    #    objs = filter_overlap(detect(im, cascade))
    #    draw_objects(im, objs, color=cv.RGB(0,255,0))
    #    eyes += len(objs)
    #faces = 0
    if debug:
        im3 = cv.CloneImage(im2)
    faces = []
    for cascade in face_cascades:#(face_cascades[0],face_cascades[-1]):
        cascade = cv.Load(cascade)
        detected_faces = detect(im2, cascade)
        faces += detected_faces
        if debug:
            for i,rect in enumerate(faces):
                rect.draw(im3, color=cv.RGB(255,16*i,16*i))
    if debug:
        image.show(im3, "Faces + Repeats")
    faces = filter_overlap(faces)
    #print (objs[1], objs[6])
    #draw_objects(im2, map(tuple, faces.keys()))
    for rect, count in faces.iteritems():
        rect.draw(im2, color=cv.RGB(255,0,0))
    #print (objs[3],objs[13])
    #draw_objects(im, filter_overlap((objs[3],objs[13])))
    
    #objs = []
    #for cascade in body_cascades:
    #    print cascade
    #    cascade = cv.Load(cascade)
    #    objs += detect(im, cascade)
    #draw_objects(im, filter_overlap(objs), color=cv.RGB(0,0,255))

    #objs = []
    #for cascade in mouth_cascades:
    #    print cascade
    #    cascade = cv.Load(cascade)
    #    objs += detect(im, cascade)
    #draw_objects(im,  filter_overlap(objs), color=cv.RGB(255,0,255))
    
    score = 0
    for face_rect, count in faces.iteritems():
        score += count * 0.25 + 0.15
    print faces

    if debug:
        image.show(im2, "Faces")
    return (im2, faces), score