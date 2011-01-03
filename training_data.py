import image
from time import time

from logger import FileLogger

class TrainingData(object):
    """Represents a training data image, for batch processing."""
    def __init__(self, imgpath, measures, bad_qualities=None, kind='bad', compare=True):
        self.imgpath, self.measures = imgpath, measures
        self.compare = compare
        self.kind = kind
        if bad_qualities is None:
            self.bad_qualities = set()
        else:
            self.bad_qualities = set(bad_qualities)
    
    def load(self, max_size=None):
        try:
            im = image.load(self.imgpath)
        except IOError:
            return None
        if max_size is not None:
            im = image.max_size(im, max_size)
        return im

    def _get_name(self, mod):
        return mod.__name__.split('.')[-1]
        
    def execute(self, im, mod, kwargs):    
        name = self._get_name(mod)
        start = time()
        mimg, score = mod.measure(im, **kwargs)
        end = time()
        return mimg, score, mod.boolean(score), end - start
        
    def process(self, max_size=None, logger=None):
        if logger is None:
            logger = FileLogger()
        im = self.load(max_size)
        if im is None:
            logger.result(self.imgpath, {'Error': "Failed to load"}, self.bad_qualities, self.compare)
            return None
        remaining = self.measures[:]
        results = {}
        # rudimentary runner that only executes ones that have the given requirements
        while len(remaining) > 0:
            still_remaining = []
            for mod in remaining:
                # verify prereqs
                abort = False
                for prereq in mod.requires_result_from:
                    if prereq not in results:
                        still_remaining.append(mod)
                        abort = True
                        break
                if abort:
                    print "Failed to meet req for:", self._get_name(mod)
                    continue
                        
                kwargs = {}
                for name in mod.requires_result_from:
                    kwargs[name] = results[name]
                results[self._get_name(mod)] = self.execute(im, mod, kwargs)
            
            msg = 'Unresolve dependencies. I have %s to satisfy %s.' % (
                ','.join(results.keys()),
                ','.join(map(
                    lambda x: self._get_name(x) + "("+','.join(x.requires_result_from)+")",
                    remaining))
            )
                
            assert len(remaining) != len(still_remaining), msg
            remaining = still_remaining
        
        logger.result(self.imgpath, results, self.bad_qualities, self.compare)

                