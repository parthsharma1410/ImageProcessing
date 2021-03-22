#importing numpy
import numpy as np
#importing pylab module of matplotlib
import matplotlib.pylab as plt
#%matplotlib inline
#Q1 Read image URL and display the image
img = plt.imread("https://2019.igem.org/wiki/images/a/a0/T--VIT_Vellore--KCL.png")
plt.imshow(img)
#Q2 Find histogram of image and plot it
hgram = np.histogram(img)
print(hgram)
#plotting the histogram
plt.hist(hgram)
plt.show()
#Histogram(pixel freq distribution)
freq = img[:, :, 0]
plt.title("Freq distribution histogram")
plt.xlabel("Value")
plt.ylabel("Freq of pixels")
plt.hist(freq)
#Q3 Apply Mask and Change the Image Color
#making mask
shape = np.shape(img)
mask = np.zeros(shape, dtype=np.uint8)
#RHS = blue
mask[:,:400] = [0, 0, 255, 255]
#LHS = red
mask[:,400:] = [255, 0, 0, 255]
plt.imshow(mask)
#Changing the image colour
mask_img = np.dot(img, [0.2, 0.5, 0.1, 0.9])
plt.imshow(mask_img)
#applying the mask we made on the color changed image
ans = 0.5*img + 0.5*mask
plt.imshow(ans)
#Q4 Convert RGB image into GrayScale Image
bnw_img = np.zeros(img.shape)
r, g, b = np.array(img[:,:,0]), np.array(img[:,:,1]), np.array(img[:,:,2])
gray = (0.2989 * r) + (0.5870 * g) + (0.1140 * b)
bnw_img = img
for i in range(3):
 bnw_img[:,:,i] = gray
plt.imshow(bnw_img)
#Q5 Crop a portion of the image
crop = img[70:500, 100:700]
plt.imshow(crop)