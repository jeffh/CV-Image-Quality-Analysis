import image
import cv

l = cv.CreateImage((256,256), cv.IPL_DEPTH_8U, 1)
cv.Set(l, 255)
u = image.new_from(l)
v = image.new_from(l)
cv.Set(u, 0)
cv.Set(v, 0)

size = cv.GetSize(l)
print size

for x in range(256):
    for y in range(size[1]):
        cv.Set2D(u, y, x, x)
        cv.Set2D(v, 255-x, min(y, 255), x)
        
image.show(u, "U")
image.show(v, "V")

rgb = image.luv2rgb(image.merge(l,u,v))
r,g,b = image.split(rgb)
#xor = image.threshold(image.Xor(u,v), 0, cv.CV_THRESH_BINARY)
xor = image.Xor(u,v)
cv.Threshold(xor, xor, 16, 255, cv.CV_THRESH_TOZERO)
image.show(rgb, "RGB")
image.show(xor, "Xor")

#cv.Sub(rgb, image.gray2rgb(image.invert(xor)), rgb)
_, sat, _ = image.split(image.rgb2hsv(rgb))
image.show(sat, 'Saturation')
#cv.Set(xor, 0, image.invert(image.threshold(sat, threshold=4)))

cv.Sub(rgb, image.invert(image.gray2rgb(xor)), rgb)

image.show(rgb, "Rgb - Xor")
arr = image.cv2array(xor)
avg_mean, avg_std = arr.mean(), arr.std()
print cv.CountNonZero(xor) / float(size[0] * size[1]), avg_mean, avg_std

cv.WaitKey()
cv.DestroyAllWindows()