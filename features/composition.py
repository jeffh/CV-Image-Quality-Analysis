import image
import cv
from features.blur import form_groups, convert_to_points, ConvexHull, draw_groups, Contours
from histogram import GrayscaleHist
from numpy import array
from contrast import average
from features import faces
from grid import Grid

def measure_focused_roi(im, roi, area, focus_points, debug=False):  
    g = Grid(cv.GetSize(im))      
    canvas = image.new_from(im)
    cv.Set(canvas, 0)
    focus_in_roi = image.And(focus_points, roi)
    if debug:
        image.show(focus_in_roi, "ROI + Focused Points")
        
    densities = []
    points = convert_to_points(focus_in_roi)
    groups = form_groups(points, estimated_size=24, iter=5)
    for group in groups:
        ch = ConvexHull(map(lambda x: (x[0], x[1]), group))
        
        ppp = ch.points_per_pixel()
        a = int(ppp * 255)
        ch.draw_filled_hull(canvas, rgb=(a,a,a))
    if debug:
        image.show(canvas, "Focused Regions in ROI")

    quadrants = g.split_in_four(canvas)
    sums = []
    for i,quad in enumerate(quadrants):
        sums.append(cv.Sum(quad)[0] / float(area/4))
    arr = array(sums)
    print arr.mean(), arr.std()
    diff = max(sums) - min(sums)
    
    return diff, arr.std()

def error_from_uniform(mean, stddev):
    return abs(mean - 128)/128.0, abs(stddev - 64)/64.0

def measure_color_roi(im, roi, area, focused_regions, debug=False):
    im = cv.CloneImage(im)
    g = Grid(cv.GetSize(im))
    
    
    """
    contours = Contours(image.threshold(focused_regions, threshold=1)).approx_poly()
    if debug:
        test = image.new_from(im)
        cv.Set(test, 0)
        for c in contours:
            i = 1
            while c:
                cv.FillPoly(test, [[c[x] for x in range(len(c))]], cv.RGB(0,64*i,0))
                c = c.v_next()
                i += 1
        #contours.draw(test, levels=9)
        image.show(test, "Test")
    """    
    #mask = image.And(image.threshold(focused_regions, threshold=1), roi)
    #
    #canvas = image.new_from(im, nChannels=1)
    #cv.Set(canvas, 0)
    #if cv.CountNonZero(mask) <= 1:
    #    return 0, 0
    #contours = Contours(image.dilate(mask)).approx_poly()
    #for c in contours:
    #    i = 1
    #    while c:
    #        cv.FillPoly(canvas, [[c[x] for x in range(len(c))]], 255)
    #        c = c.v_next()
    #        i += 1
    #mask = image.Or(mask, canvas)
    #if debug:
    #    image.show(mask, "MASK")
    #        
    #cv.Set(im, 0, image.invert(mask))
    cv.Set(im, 0, image.invert(roi))
    
    #area = cv.CountNonZero(image.threshold(im, threshold=1))
    
    if debug:
        image.show(g.draw(im,thickness=2), "Image + ROI + Focus point mask")

    scores = []
    im = image.rgb2gray(im)
    #canvas = image.And(plane, roi)
    quadrants = g.split_in_four(im)
    hist = []
    for q,quad in enumerate(quadrants):
        #scores.append(cv.Sum(quad)[0] / float(area/4))
        h = GrayscaleHist(value_range=(1,255)).use_image(quad)
        #image.show(h.to_img(), ['gray', 'red','green','blue'][i] + ' in ' + str(q))
        hist.append(h.to_array())
    scores = []
    excluded_points = set([(2, 1), (3, 0)])
    for i,h1 in enumerate(hist):
        for j,h2 in enumerate(hist):
            if i <= j or (i,j) in excluded_points:
                continue
            h = abs(h2-h1)
            ht = GrayscaleHist(value_range=(0,255)).use_array_as_hist(h)
            scores.append((h[5:].mean(), h[5:].std()))
    means = max([x[0] for x in scores])        
    stddevs = max([x[1] for x in scores])
    return means/255.0, stddevs/255.0

def measure_saturation(im, debug=False):
    _, sat, _ = image.split(image.rgb2hsv(im))
    arr = image.cv2array(sat)
    return arr.mean(), arr.std()
    
def measure_contrast(im, debug=False):
    h = GrayscaleHist(bins=64).use_image(image.rgb2gray(im))
    return h.stddev()
    
def boolean((focused, contrast, saturation, face_score)):
    # focus = maximum different between one of four quads interms of # of focus pixels
    #focus, (means, stddev), face_score = score
    #return focus < 0.1 or (means > 0.05 or stddev > 0.025) #or faces.boolean(face_score)
    focused_mean_diff, focused_std = focused
    sat_mean, sat_std = saturation
    if not faces.boolean(face_score):
        return False
    if contrast > 48:
        return False
    return focused_mean_diff - focused_std * 2 < 0.2 or sat_mean + sat_std <= 96
    
requires_result_from = ['blur']
def measure(im, blur, noise=None, debug=False):
    focus_points = blur[0]
    #is_noisy = noise[2]

    size = cv.GetSize(im)
    npixels = size[0] * size[1]
    
    #if focused_regions is None:
    #    focused_regions = image.new_from(im)
    #    cv.Set(focused_regions, 0)
    #    groups = form_groups(focus_points,
    #        estimated_size=min(max(int(len(npixels) / 1000), 2), 15))
    #    #groups = form_groups(points, threshold=max(cv.GetSize(im))/16)
    #    #print 'groups', len(groups)
    #    draw_groups(groups, focused_regions)
    
    im2 = cv.CloneImage(im)
    g = Grid(cv.GetSize(im2))
    if debug:
        image.show(g.draw(im2), "Image with Grid + ROI")
    
    roi = image.new_from(im, nChannels=1)
    cv.Set(roi, 0)
    #g.draw_lines(roi, thickness=int(max(min((size[0] + size[1]) * 1/100.0, 255), 1)))
    g.draw_regions(roi)
    area = cv.Sum(roi)[0]
    
    (_, face_rects), face_score = faces.measure(im)
    face_block = image.new_from(im, nChannels=1)
    cv.Set(face_block, 0)
    for r in face_rects:
        r.draw(face_block, color=cv.RGB(255,255,255), thickness=cv.CV_FILLED)
    
    if debug:
        face_roi = cv.CloneImage(im)
        cv.Set(face_roi, 0, image.invert(roi))
        cv.Set(face_roi, 0, image.invert(image.threshold(face_block, threshold=1)))
        
        image.show(face_block, "Faces in Binary")
        image.show(g.draw(face_roi), "Face + ROI")
        
    return (im, (
         measure_focused_roi(im, roi, area, focus_points, debug),
         #measure_color_roi(im, roi, area, focus_points, debug),
         measure_contrast(im, debug),
         measure_saturation(im, debug),
         faces.measure(im, debug)[1],
    ))