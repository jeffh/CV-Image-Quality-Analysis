#!/usr/bin/python2.6
import sys
import image
import cv
import os
from windows import (EdgeThresholdTweaker, DerivativeTweaker, \
    ColorHistograms, HistogramWindow)
#from constants import edge_threshold
import papers
from features import noise, blur, contrast, composition, faces
from training_data import TrainingData
from logger import FileLogger, HtmlLogger, MultiLogger
from optparse import OptionParser

#import matplotlib.pyplot as plt
measurements = (
    contrast,
    noise,
    blur,
    composition,
    #faces, # => under composition
)
max_size = (640, 480) # set to None to use original image sizes (can take awhile!)

# good - bad_qualities = possible bad qualities that might be marked
california_night = TrainingData('good/Another Beautiful California Night.jpg',
    measures=measurements, kind='good',
)
bnw_horse = TrainingData('good/AP McCoy Black & White Horse Racing Photo.jpg',
    measures=measurements, kind='good', #bad_qualities=['grayscale'],
)
dreamer = TrainingData('good/Beautiful Dreamer, Awake Unto Me.jpg',
    measures=measurements, kind='good',# bad_qualities=['composition']
)
tiger = TrainingData('good/Beautiful Face.jpg',
    measures=measurements, kind='good',
)
bricks = TrainingData('good/Climbing the Bricks Factory.jpg',
    measures=measurements, kind='good',
)
grass = TrainingData('good/It is time to make each moment beautiful.jpg',
    measures=measurements, kind='good',#bad_qualities=['blur']
)
rule_of_thirds = TrainingData('good/Rule of thirds.jpg',
    measures=measurements, kind='good',#bad_qualities=['blur']
)
china_family = TrainingData('good/china_family.jpg',
    measures=measurements, kind='good',
)
fair = TrainingData('good/Long_exposure_at_the_fair.jpg',
    measures=measurements, kind='good',#bad_qualities=['blur']
)
cat = TrainingData('good/portrait of tracy II.jpg',
    measures=measurements, kind='good',#bad_qualities=['blur']
)
dof_ground = TrainingData('good/dof_ground.jpg',
    measures=measurements, kind='good',#bad_qualities=['blur']
)
cloth = TrainingData('good/cloth.jpg',
    measures=measurements, kind='good',
)

# poor
blurry = TrainingData('poor/Blurry_men_climbing_stairs.jpeg', measures=measurements,
    bad_qualities=['noise', 'blur']
)
ferris_wheel = TrainingData('poor/Ferris wheel at night.jpg', measures=measurements,
    bad_qualities=['noise', 'composition', 'blur'] # i'm on the fence on noise
)
room = TrainingData('poor/mobys house.jpg', measures=measurements,
    bad_qualities=['noise', 'blur']
)
indiana = TrainingData('poor/Northwestern Indiana.jpg', measures=measurements,
    bad_qualities=['noise', 'contrast', 'composition', 'blur']
)
gates = TrainingData('poor/overexposed gates.jpg', measures=measurements,
    bad_qualities=['contrast']
)
seaside = TrainingData('poor/overexposed seaside.jpg', measures=measurements,
    bad_qualities=['contrast']
)
subway = TrainingData('poor/Overexposed U-bahn.jpg', measures=measurements,
    bad_qualities=['contrast', 'blur']
)
corner = TrainingData('poor/That Corner.jpg', measures=measurements,
    bad_qualities=['contrast']
)
transformer = TrainingData('poor/transformer blurry 1869.jpeg', measures=measurements,
    bad_qualities=['noise', 'blur', 'contrast']
)
street = TrainingData('poor/underexposed.jpg', measures=measurements,
    bad_qualities=['noise', 'contrast']
)
china_hotel = TrainingData('poor/china_hotel.jpg', measures=measurements,
    bad_qualities=['blur', 'noise']
)
china_blurry = TrainingData('poor/china_blurry.jpg', measures=measurements,
    bad_qualities=['blur', 'noise']
)
china_noise = TrainingData('poor/china_noise.jpg', measures=measurements,
    bad_qualities=['contrast', 'noise', 'blur']
)

#li = TrainingData('li/poor/Syracuse U. Visit 028.jpg', measures=measurements,
#    bad_qualities=['blur']
#)
#
#li2 = TrainingData('li/good/DSC00144.JPG', measures=measurements,)
#
#li3 = TrainingData('li/poor/ZXCVBNM 028.jpg', measures=measurements,
#    bad_qualities=['blur', 'noise']
#)
#
#li4 = TrainingData('li/poor/IMG_0795.JPG', measures=measurements,
#)
#
#li5 = TrainingData('li/good/Syracuse U. Visit 005.jpg', measures=measurements,
#    bad_qualities=['contrast']
#)

#chow = TrainingData('chow/good/DSC_0291.JPG', measures=measurements)

def load_image(image_path):
    root = os.path.basename(image_path)
    return TrainingData(image_path,
        measures=measurements, kind="good" if root.find('good') != -1 else "bad",
        compare=False)

def load_images_from(dirname):
    objs = []
    for root,dirs,files in os.walk(os.path.join('data', dirname)):
        for f in files:
            name = f.lower()
            if not (name.endswith('.jpeg') or name.endswith('.jpg')):
                continue
            objs.append(load_image(os.path.join(root, f).replace('data/', '')))
    return objs

li_dir = load_images_from('li')
chow_dir = load_images_from('chow')
china_dir = load_images_from('/Users/jeff/Desktop/china-day6/full')


def process(process_dir=None):
    if process_dir is None:
        process_dir = [v for v in globals().values() if isinstance(v, TrainingData)]
    
    logger = MultiLogger(
        FileLogger(),
        HtmlLogger(open('output.html', 'w+'), os.path.abspath('thumb'))
    )
    max_size = None#(800, 600)
    for obj in process_dir:
        if obj.kind == 'good':
            obj.process(max_size=max_size, logger=logger)
    for obj in process_dir:
        if obj.kind == 'bad':
            obj.process(max_size=max_size, logger=logger)
        

from windows import CornerTweaker
def main(progname, *args):

    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename", default=None,
        help="analyze a given FILE ending in .jpg or .jpeg", metavar="FILE")
    parser.add_option("-i", "--imageset", dest="imageset", default=None,
        help="Runs on a predefined set of algorithms (li,chow,china,custom)")
    parser.add_option("-d", "--debug", dest="debug", action="store_true", default=False,
        help="Enable visual debugging.")
    parser.add_option("-t", "--type", dest="type", default="all",
        help="Specifies the type of feature to debug. Defaults to all.")

    (options, args) = parser.parse_args(list(args))
    
    if options.imageset:
        if options.imageset == 'li':
            process(li_dir)
            return 0
        elif options.imageset == 'chow':
            process(chow_dir)
            return 0
        elif options.imageset == 'china':
            process(china_dir)
            return 0
        elif options.imageset == 'custom':
            process()
            return 0
            
    if not options.filename:
        print "Please specify a file (--file) or image set (--imageset)."
        return 1
    
    if not options.debug:
        process([load_image(options.filename)])
        return 0
        
    if options.filename.startswith('data/'):
        options.filename = options.filename[len('data/'):]
    
    tdata = load_image(options.filename)
    kind = options.type.lower()
    
    size = None #(320,240,'crop') # (0.5, 0.5, 'resize-p')
    if size is None:
        im = tdata.load()
    elif size[-1] == 'crop':
        im = image.random_cropped_region(tdata.load(), size[:2])
    elif size[-1] == 'resize':
        im = tdata.load(size[:2])
    elif size[-1] == 'resize-p':
        im = image.resize(tdata.load(), by_percent=size[:2])
    else:
        raise TypeError, "Invalid image sizing type."
        
    image.show(im, "Image")
    #l,u,v = image.split(image.rgb2luv(im))
    ##cv.Set(l, 128)
    ##cv.EqualizeHist(l, l)
    ##cv.EqualizeHist(u, u)
    ##image.show(image.luv2rgb(image.merge(l,u,v)), "test")
    #s = cv.GetSize(im)
    #t = image.absDiff(u,v)
    #image.show(t, "test")
    #print "Test Score:", cv.CountNonZero(t) / float(s[0] * s[1])
    ##image.show(image.threshold(image.And(u,v), threshold=1), "LUV")

    # noise
    if kind in ('all','noise'):
        noise_img, score = noise.measure(im, debug=True)
        #image.show(noise_img, "Noise Result")
        print 'Noise Score:', score, noise.boolean(score)
    
    # contrast
    if kind in ('all','contrast'):
        contrast_img, score = contrast.measure(im, debug=True)
        #image.show(contrast_img, "Contrast Result")
        print 'Contrast Score:', score, contrast.boolean(score)
    
    # blur
    if kind in ('all','blur','composition'): 
        focused, score = blur.measure(im, debug=kind in ('all','blur'))
        #image.show(focused,  "Blur Result")
        print 'Blur Score:', score, blur.boolean(score)
    
    # composition
    if kind in ('all','composition'):
        composition_img, score = composition.measure(im,
            (focused,score, blur.boolean(score)), debug=True)
        print 'Composition Score:', score, composition.boolean(score)
        
    if kind in ('faces',):
        result, score = faces.measure(im,debug=True)
        print "Face Score:", score, faces.boolean(faces)
    
    #win = CornerTweaker(im)
    #win.show()
    
    #_, sat, _ = image.split(image.rgb2hsv(im))
    #arr = image.cv2array(sat)
    #print arr.mean(), arr.std()
    
    # faces
    #im, score = faces.measure(im, debug=True)
    #print score, faces.boolean(score)
    
    # composition
    #noise_img, score = noise.measure(im, debug=False)
    ##n = (noise_img, score, noise.boolean(score))
    #hulls, score = blur.measure(im, debug=False)
    #b = (hulls, score, blur.boolean(score))
    #cimg, score = composition.measure(im, b, debug=True)
    #print score, composition.boolean(score)
    
    # BLUR
    #from time import time
    #start = time()
    ##im2 = image.threshold(image.laplace(im), threshold=75, type=cv.CV_THRESH_TOZERO)
    #hulls, score = blur.measure(im, debug=True)
    ##blur_img, score = blur.measure(im, debug=True)
    #end = time()
    #print "Time:", (end - start), "seconds"
    #image.show(im,  "image")
    ##image.show(noise_img, "Noise Image")
    #print score, blur.boolean(score)
    
    
    #CONTRAST
    
    #_, score = contrast.measure(im, debug=True)
    #image.show(im, "Image")
    #print score, contrast.boolean(score)

    """
    
    #BLUR
    
    #im2 = image.threshold(image.laplace(im), threshold=75, type=cv.CV_THRESH_TOZERO)
    im3, score = blur.measure(im, debug=True)
    image.show(im,  "image")
    image.show(im3, "Focus Mask")
    print score, blur.boolean(score)
    #plt.show()
    """

    
    #NOISE
    
    #noise_img, score = noise.measure(im, debug=True)
    #image.show(noise_img, "Noise")
    #print score, noise.boolean(score)
    
    
    """
    #hwin = ColorHistograms(im)
    #hwin.show()
    hwin = HistogramWindow(image.rgb2gray(im))
    hwin.show()
    
    print cv.GetSize(im), cv.GetSize(im2)
    print 'blur', papers.blurry_histogram(im)
    #print papers.blurry_histogram(im2)
    
    wind = DerivativeTweaker(im, title="image derivative")
    wind.show()
    
    win = EdgeThresholdTweaker(im, title="image edges")
    win.show(50)#edge_threshold(im))
    
    #win2 = EdgeThresholdTweaker(im2, title="image resized edges")
    #win2.show(edge_threshold(im2))
    """
    cv.WaitKey()
    cv.DestroyAllWindows()
    return 0

if __name__ == '__main__':
    sys.exit(main(*sys.argv))