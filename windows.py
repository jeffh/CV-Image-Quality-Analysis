import cv
import image

class ArgTweakerWindow(object):
    """Handles the display of images and tweaking its values.
     - im: the original image to process
     - render: the function to process im and render_kwargs. Should return a 
               new image.
     - render_kwargs: (optional) a dict of function kwargs to pass to
               the render function.
     - title: (optional) the title of the window. One will be uniquely created
              if not specified.
    """
    def __init__(self, im, render, render_kwargs={}, title=None):
        self.title, self.im, self._render = title, im, render
        self.render_args = render_kwargs
        self._tb = []
        self.reset_args()
        
    def rename_arg(self, name):
        """Renames an arg that's more like python function parameter names.
        'Threshold Value' => 'threshold_value'
        """
        return name.lower().replace(' ', '_')
    
    def num_arg_steps(self, name):
        """Returns number of steps for a trackbar for a particular argument."""
        min, max, step = self.render_args[name][:3]
        return int((max - min) / step)
        
    def mid_arg_value(self, name):
        """Returns the median value of the ranges of values the trackbar
        can take.
        """
        return self.arg_value(name, int(self.num_arg_steps(name) / 2))
        
    def arg_value(self, name, trackbar_value):
        """Returns the value at a given trackbar location."""
        min, max, step = self.render_args[name][:3]
        return min + step * trackbar_value
    
    def reset_args(self):
        """Resets all the known values set by the trackbars to their default
        values. Does not update the trackbar.
        """
        self.args = {}
        for name,tupl in self.render_args.iteritems():
            if len(tupl) > 3:
                print tupl
                self.args[self.rename_arg(name)] = tupl[-1]
            else:
                self.args[self.rename_arg(name)] = self.mid_arg_value(name)
    
    def show(self):
        """Creates a new window and displays it, along with trackbars to
        tweak any given arguments.
        """
        if self.title is None:
            self.title = "Image - " + str(id(self))
        cv.NamedWindow(self.title, flags=0)
        self.reset_args()
        for name,arg_range in self.render_args.iteritems():
            def execute(name, total):
                min = self.render_args[name][0]
                steps = self.render_args[name][2]
                self._tb.append(cv.CreateTrackbar(
                    name, self.title,
                    (self.args[name] - (min - 1)) / steps - 1,
                    total,
                    lambda x: self.update_arg(name, x)))
            execute(name, self.num_arg_steps(name))
        self.update()
    
    def render(self):
        """Shorthand to calling the function passed into the constructor.
        Returns the new image to display
        """
        return self._render(self.im, **self.args)
        
    def update(self):
        """Updates the window with the rendered image."""
        im = self.render()
        size = cv.GetSize(im)
        cv.ShowImage(self.title, im)
        cv.ResizeWindow(self.title, size[0], size[1] + len(self.args) * 35)
        
    def update_arg(self, name, x):
        """This is invoked by each trackbar.
        """
        new_value = self.arg_value(name, x)
        if new_value != self.args[self.rename_arg(name)]:
            print 'set', repr(name), new_value
        self.args[self.rename_arg(name)] = new_value
        self.update()
        return self.args
    
    def destroy(self):
        """Destroys the window associated with this instance."""
        cv.DestroyWindow(self.title)
        
class EdgeThresholdTweaker(ArgTweakerWindow):
    """An edge threshold tweaker window."""
    def __init__(self, im, title=None):
        super(EdgeThresholdTweaker, self).__init__(im, self.draw, {
            'threshold': [1, 300, 1, 50],
        }, title)
        
    def show(self, default_threshold=50):
        self.render_args['threshold'][3] = default_threshold
        super(EdgeThresholdTweaker, self).show()
    
    def draw(self, im, threshold):
        new_im = image.new_from(im)
        edges = image.edges(im, threshold, threshold*3, 3)
        cv.SetZero(new_im)
        cv.Copy(im, new_im, edges)
        size = cv.GetSize(im)
        print cv.CountNonZero(image.threshold(edges)) / float(size[0] * size[1])
        #cv.Dilate(new_im, new_im)
        #cv.Erode(new_im, new_im)
        return new_im

class DerivativeTweaker(ArgTweakerWindow):
    """An image derivative tweaker window."""
    def __init__(self, im, title=None):
        super(DerivativeTweaker, self).__init__(im, self.draw, {
            'order': [1, 8, 1, 1],
        })
        self.im = im
        self.title = title
        
    def draw(self, im, order):
        return image.sobel(cv.CloneImage(im), xorder=order, yorder=order)

        
class CornerTweaker(ArgTweakerWindow):
    """An edge threshold tweaker window."""
    def __init__(self, im, title=None):
        #corners(im, max_corners=100, quality=0.1, min_dist=5, block_size=3, use_harris=False,
        #    mask=None, k=0.04):
        super(CornerTweaker, self).__init__(im, self.draw, {
            'max_corners': [100, 10000, 10, 100],
            'quality': [0.05, 1.00, 0.05, 0.1],
            'min_dist': [1, 100, 1, 5],
            'block_size': [1, 100, 1, 3],
            'use_harris': [0, 1, 1, 0],
        }, title)
    
    def draw(self, im, **kwargs):
        new_im = cv.CloneImage(im)
        corners = image.corners(im, **kwargs)
        image.draw_points(new_im, corners)
        return new_im

class HistogramWindow(object):
    """Displays a histogram for a given image."""
    def __init__(self, im, title=None, hist_size=256, color=cv.ScalarAll(0)):
        ranges = [ [0, hist_size] ]
        self.hist_size = hist_size
        self.color = color
        self.im = im
        self.hist_image = cv.CreateImage((320, 200), 8, 3)
        self.hist = cv.CreateHist([hist_size], cv.CV_HIST_ARRAY, ranges, 1)
        self.title = title
    
    def update(self, im=None):
        if im is not None:
            self.im = im

        cv.CalcArrHist([self.im], self.hist)
        (min_value, max_value, _, _) = cv.GetMinMaxHistValue(self.hist)
        cv.Scale(self.hist.bins, self.hist.bins, float(self.hist_image.height) / max_value, 0)

        cv.Set(self.hist_image, cv.ScalarAll(255))
        bin_w = round(float(self.hist_image.width) / self.hist_size)

        for i in range(self.hist_size):
            cv.Rectangle(self.hist_image, (int(i * bin_w), self.hist_image.height),
                         (int((i + 1) * bin_w), self.hist_image.height - cv.Round(self.hist.bins[i])),
                         self.color, -1, 8, 0)

        cv.ShowImage(self.title, self.hist_image)
        
    def show(self):
        if self.title is None:
            self.title = "Histogram-"+str(id(self))
        cv.NamedWindow(self.title, 0)
        self.update()

class ColorHistograms(object):
    def __init__(self, im, title=None):
        assert im.nChannels == 3
        self.title = title
        ims = image.split(im)
        if self.title is None:
            self.title = "Histogram-" + str(id(im))+ "-"
        self.histograms = (
            HistogramWindow(ims[0], self.title+"red", color=cv.Scalar(255,0,0)),
            HistogramWindow(ims[1], self.title+"green", color=cv.Scalar(0,255,0)),
            HistogramWindow(ims[2], self.title+"blue", color=cv.Scalar(0,0,255)),
        )
    
    def show(self):
        for h in self.histograms:
            h.show()
            
class GrayRGBWindow(ArgTweakerWindow):
    def __init__(self, images, title=None):
        assert len(images) == 4, "Requries (grayscale, r, g, b)"
        super(GrayRGBWindow, self).__init__(images, self.draw, {
            'layers': [0, len(images), 1, 0],
        }, title)

    def show(self, layer=0):
        self.render_args['layers'][3] = layer
        super(GrayRGBWindow, self).show()

    def draw(self, images, layers):
        if layers == 0:
            gray, r,g,b = images
            im = cv.CloneImage(r)
            for i in (g,b):
                im = image.add(im, i)
            white = image.new_from(gray)
            cv.Set(white, (0,0,0))
            cv.Set(white, (255,255,255), image.invert(image.threshold(im, threshold=1)))
            im = image.Or(image.blend(im, gray), white)
        else:
            im = cv.CloneImage(images[layers-1])
            if layers != 1:
                white = image.new_from(images[0])
                cv.Set(white, (0,0,0))
                cv.Set(white, (255,255,255), image.invert(image.threshold(im, threshold=1)))
                im = image.Or(im, white)
        return im
        