import image
import cv
import random
import noise
import math
from numpy import array

# saturation is all ZERO when image is grayscale!
def get_focus_points(im,debug=False):
    edges = image.dilate(image.auto_edges(im))
    #d = 1
    #sobel = image.sobel(im, xorder=d, yorder=d)
    sobel = image.laplace(im)
    hsv = image.rgb2hsv(im)

    focused = image.And(sobel, edges)
    if im.nChannels == 3:
        hue, saturation, value = image.split(hsv)
        saturation = image.dilate(saturation)
        focused = image.And(focused, saturation)
    focused = image.threshold(image.dilate(focused), threshold=32)
    
    if debug:
        image.show(edges, "1. Edges")
        image.show(sobel, "2. Sobel")
        if im.nChannels == 3:
            image.show(saturation, "3. Saturation")
    return focused
    
def convert_to_points(im):
    points = []
    size = cv.GetSize(im)
    for x in xrange(size[0]):
        for y in xrange(size[1]):
            if im[y, x] > 0:
                points.append((x, y))
    return points
    
"""
# too slow, too inefficient
from hcluster import pdist, linkage, to_tree, dendrogram, centroid
def form_groups(points, threshold=5, min_size=0.025):
    "Clusters points based on a given threshold. If the cluster's dist > threshold, it is split."
    print "Threshold =", threshold
    points = array(points)
    #Y = pdist(points)
    print "centroids"
    Z = centroid(points)
    print "dendrogram"
    dendrogram(Z)
    print "to_tree"
    R = to_tree(Z)
    print "get_count"
    total = R.get_count()
    clusters = [R]
    seen = set()
    min_size = max(int(total * min_size), 4)
    
    while 1:
        if clusters[0] in seen:
            break
        node = clusters.pop(0)
        diff = abs(node.get_left().get_count() - node.get_right().get_count())
        should_split = not node.is_leaf() and node.dist > threshold
        should_split = should_split and total > min_size and \
            node.get_left().get_count() > min_size and \
            node.get_right().get_count() > min_size
        if not should_split:
            clusters.append(node)
            seen = seen.union([node])
        else:
            clusters.extend((node.get_left(), node.get_right()))
    print 'clusters', len(clusters)
    def get_ids(c, accum=None):
        if c.is_leaf():
            return [c.id]
        l = []
        l.extend(get_ids(c.get_left()))
        l.extend(get_ids(c.get_right()))
        return l
    return [get_ids(c) for c in clusters]
"""
from scipy.cluster.vq import whiten, kmeans2, vq
def form_groups(points, estimated_size=10, iter=1):
    if len(points) < 1:
        return []
    points = array(points)
    centroids, variance = kmeans2(points, estimated_size, iter=iter, minit='points')
    group_indicies, dist = vq(points, centroids)
    group = {}
    for i,index in enumerate(group_indicies):
        if index not in group:
            group[index] = []
        group[index].append(points[i])    
    return group.values()

def draw_groups(groups, im=None):
    
    hulls = []
    for group in groups:
        ch = ConvexHull([(g[0], g[1]) for g in group])#map(lambda x: (x[0], x[1]), group))
        #ch = ConvexHull(map(lambda x: points[x], group))
        hulls.append(ch)
        
        ppp = ch.points_per_pixel()
        #if ch.area() >= min_area: # not a 3x3 region
        #    densities.append(ppp)
        a = int(ppp * 255)
        if im:
            ch.draw_filled_hull(im, rgb=(a,a,a))#r(255),r(255),r(255)))
        
        
        #if debug and ch.area() < min_area:
        #    ch.draw_centroid(focused_regions, rgb=(255,0,0))
        
        #ch.draw_hull(focused_regions, rgb=(r(255),r(255),r(255)))

        #ch.draw_points(focused_regions, rgb=(r(255),r(255),r(255)))
    return hulls

class Contours(object):
    def __init__(self, im):
        self.im = im
        self.storage = cv.CreateMemStorage(0)
        self.contours = cv.FindContours(im, self.storage, cv.CV_RETR_TREE,
            cv.CV_CHAIN_APPROX_SIMPLE)
            
    def approx_poly(self):
        self.contours = cv.ApproxPoly(self.contours, self.storage,
            cv.CV_POLY_APPROX_DP, 3, 1)
        return self
    
    def draw(self, im, levels=3, thickness=3, external_rgb=(255,0,0), internal_rgb=(0,255,0)):
        # draw contours in red and green
        cv.DrawContours (im, self.contours,
            cv.RGB(*external_rgb), cv.RGB(*internal_rgb),
            levels, thickness, cv.CV_AA)
            
    def __iter__(self):
        start = contour = self.contours
        while contour:
            yield contour
            contour = contour.h_next()
        self.contours = start
    
class ConvexHull(object):
    def __init__(self, points):
        # compute the convex hull
        self.boundary_points = points[:]
        self.points = points
        self.storage = cv.CreateMemStorage(0)
        self.hull = cv.ConvexHull2(self.points, self.storage, cv.CV_CLOCKWISE, 1)
        self._centroid = None
    
    def __contains__(self, point):
        return cv.PointPolygonTest(self.boundary_points, point, 0) >= 0
    
    def draw_points(self, im, rgb=(255,0,0)):    
        for pt in self.points:
            cv.Circle (im, pt, 2, cv.RGB(*rgb),
                         cv.CV_FILLED, cv.CV_AA, 0)
    
    def draw_filled_hull(self, im, rgb=(255,255,255)):
        cv.FillPoly(im, [self.hull], cv.RGB(*rgb), cv.CV_AA)
    
    def draw_hull(self, im, rgb=(0,255,0)):
        cv.PolyLine(im, [self.hull], 1, cv.RGB(*rgb), 1, cv.CV_AA)
        
    def draw_centroid(self, im, rgb=(0,0,255)):    
        cv.Circle (im, self.centroid(), 2, cv.RGB(*rgb),
                     cv.CV_FILLED, cv.CV_AA, 0)
                     
    def area(self):
        return max(abs(cv.ContourArea(self.hull)), 0.00001)
        
    def points_per_pixel(self):
        return len(self.points) / self.area()
        
    def centroid(self):
        if self._centroid is None:
            sum = [0, 0]
            for p in self.points:
                sum[0] += p[0]
                sum[1] += p[1]
            s = float(len(self.points))
            self._centroid = int(sum[0] / s), int(sum[1] / s)
        return self._centroid

def pixel_remove(im):
    p = im[1,1]
    if p == 0:
        return 0
    c = 0
    if im[0,0] > 0 and im[2,2] > 0:
        c += 1
    if im[1,0] > 0 and im[1,2] > 0:
        c += 1
    if im[2,0] > 0 and im[0,2] > 0:
        c += 1
    if im[0,1] > 0 and im[2,1] > 0:
        c += 1
    if c > 2:
        return 0
    return c > 0
        
def remove_useless_points(im):
    return image.neighbor_map(im, pixel_remove)

# ri_threshold = 0.2 is too weak for large images
def boolean(score, ri_threshold=0.1, d_threshold=0.2, d_high_threshold=0.7):
    # the focus regions should cover 15%+ of the screen
    # the density of the regions should be > 20% point/pixel
    region_image_ratio, density, density_std, saturation_score, light_score = score
    if density > d_high_threshold:
        return False
    if light_score > 0.5:
        return True
    return (region_image_ratio <= ri_threshold and \
        #(density < d_threshold or density_std < d_threshold)) \
        (density - density_std < d_threshold)) \
        or saturation_score > 0.8
    #return score < d_threshold

requires_result_from = [] # noise perhaps???
# was using Agglomerative Clustering, but too slow for points > ~5000
def measure(im, debug=False):
    size = cv.GetSize(im)
    npixels = size[0] * size[1]
    #print 'np', npixels
    
    
    focused = get_focus_points(im, debug)
    points = convert_to_points(focused)
    
    if debug:
        print "\t"+str(len(points)), '/', npixels, '=', len(points) / float(npixels)
        print "\tlen(points) =", len(points)
        image.show(focused, "4. Focused Points")

    saturation_score = 0
    if not image.is_grayscale(im):
        edges = image.auto_edges(im)
        _, saturation, _ = image.split(image.rgb2hsv(im))
        if debug:
            image.show(saturation, "5. Saturation")
        #saturation = image.laplace(image.gaussian(saturation, 3))
        saturation = image.invert(saturation)
        mask = image.invert(image.threshold(im, threshold=16))
        if debug:
            image.show(saturation, "5.3. Laplace of Saturation")
        cv.Set(saturation, 0, mask)
        cv.Set(saturation, 0, focused)
        if debug:
            image.show(mask, "5.6. Mask(focused AND invert(threshold(im, 16)))")
            image.show(saturation, "6. Set(<5.3>, 0, <5.6>)")

        saturation_score = cv.Sum(saturation)[0] / float(npixels * 32)
        print "\tSaturation Score:", saturation_score
        
    # light exposure
    h,s,v = image.split(image.rgb2hsv(im))
    if debug:
        image.show(h, "7. Hue")
        image.show(s, "7. Saturation")
        image.show(v, "7. Value")
    diff = cv.CloneImage(v)
    cv.Set(diff, 0, image.threshold(s, threshold=16))
    diff = image.dilate(diff, iterations=10)
    if debug:
        thres_s = image.threshold(s, threshold=16)
        image.show(thres_s, "8.3. Mask(threshold(<7.Saturation>, 16))")
        image.show(diff, "8.6. Dilate(Set(<7.Value>, 0, <8.3>), 10)")

    cdiff = cv.CountNonZero(diff)
    if cdiff > 0 and cdiff / float(npixels) > 0.01:
        test = cv.CloneImage(v)
        cv.Set(test, 0, image.invert(diff))
        s = cv.Sum(test)[0] / float(cdiff * 255)
        if debug:
            print '\tLight Exposure Score:', s
    else:
        s = 0
        
    if image.is_grayscale(im):
        return focused, (1, 1, 1, saturation_score, s)
    
    # we want to short circuit ASAP to avoid doing KMeans 50% of the image's pixels
    if len(points) > npixels/2:
        return focused, (1, 1, 1, saturation_score, s)

    # we're so blurry we don't have any points!
    if len(points) < 1:
        return focused, (0, 0, 0, saturation_score, s)
    
    if debug:
        im2 = cv.CloneImage(im)
    focused_regions = image.new_from(im)
    cv.Set(focused_regions, 0)
    
    r = lambda x: random.randrange(1, x)
    groups = form_groups(points,
        estimated_size=min(max(int(len(points) / 1000), 2), 15))
    #groups = form_groups(points, threshold=max(cv.GetSize(im))/16)
    #print 'groups', len(groups)
    hulls = draw_groups(groups, focused_regions)
    focused_regions = image.threshold(focused_regions, threshold=32, type=cv.CV_THRESH_TOZERO)
    min_area = npixels * 0.0005
    densities = [h.points_per_pixel() for h in hulls if h.area() >= min_area]
    
    if debug:    
        #image.show(focused, "Focused Points")
        image.show(focused_regions, "9. Focused Regions from <4>")
        cv.Sub(im2, image.gray2rgb(image.invert(image.threshold(focused_regions, threshold=1))), im2)
        image.show(im2, "10. threshold(<9>)")
    
    
    focused_regions = image.rgb2gray(focused_regions)
    
    densities = array(densities)
    c = cv.CountNonZero(focused_regions)
    c /= float(npixels)
    
    score = (c, densities.mean(), densities.std(), saturation_score, s)
    
    return focused, score