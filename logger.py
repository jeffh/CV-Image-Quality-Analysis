import sys, os
import image

class FileLogger(object):
    def __init__(self, handle=None):
        if handle is None:
            handle = sys.stdout
        self.handle = handle
    
    def write(self, *message, **kwargs):
        self.handle.write(' '.join(map(str, message)) % kwargs)
        
    def writeln(self, *message, **kwargs):
        self.write(*message, **kwargs)
        self.handle.write("\n")
        
    def result(self, image_path, results, expected, compare=True):
        self.writeln(image_path + ":")
        for name,result in results.iteritems():
            self.writeln("\t%(e)s%(name)s => %(score)r %(b)s (%(sec)s secs)",
                name=name, score=result[1],
                b="Yes" if result[2] else "No",
                e="* " if compare and name in expected else "",
                sec=result[3])
    
    def close(self):
        self.handle.close()
    
class HtmlLogger(FileLogger):
    def __init__(self, handle=None, resized_dir=None):
        super(HtmlLogger, self).__init__(handle)
        self.resized_dir = resized_dir
        self.write("""<html><head><title>Log</title>
<style type="text/css">
.bad { background: red; }
body { font-family: sans-serif; }
td { border: 1px solid gray }
</style>
</head>
<body>
<table>
<tr>
<th>Image</th>
<th>Metric</th>
<th>Score</th>
<th>Result?</th>
<th>Expected?</th>
<th>Compute Time</th>
</tr>
""")
    
    def result(self, image_path, results, expected, compare=True):
        fullpath = path = os.path.abspath(os.path.join('data', image_path))
        if self.resized_dir is not None:
            im = image.max_size(image.load(path), (320, 240))
            path = os.path.join(self.resized_dir, os.path.basename(image_path))
            print os.path.dirname(image_path)
            image.save(im, path)
            del im
        self.write('<tr>')
        self.write('<td rowspan="%(len)d"><a href="%(fullpath)s"><img src="%(path)s" width="320" /><br />%(imgpath)s</a></td>',
            fullpath=fullpath, path=path, imgpath=image_path, len=len(results))
        for i,pair in enumerate(results.iteritems()):
            name,result = pair
            if i != 0:
                self.write("</tr><tr>")
            self.write('<td>%(name)s</td><td>%(score)r</td>',
                name=name, score=result[1])
            self.write('<td class="%(cls)s">%(confirmed)s</td><td>%(expected)s</td>',
                cls="bad" if compare and result[2] != (name in expected) else "good",
                confirmed="Yes" if result[2] else "",
                expected="Yes" if name in expected else "")
            self.write('<td class="%(cls)s">%(timing).2f second(s)</td>',
                cls="bad" if compare and result[3] > 10 else "",
                timing=result[3])
        self.write("</tr>")
     
    def close(self):
        self.write("</table></body></html>")
        super(HtmlLogger, self).close()

class MultiLogger(object):
    def __init__(self, *loggers):
        self.loggers = loggers
    
    def result(self, image_path, results, expected, compare=True):
        for l in self.loggers:
            l.result(image_path, results, expected, compare)
    
    def close(self):
        for l in self.loggers:
            l.close()