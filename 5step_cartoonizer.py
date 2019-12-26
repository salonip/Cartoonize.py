import cv2
import numpy as np
import imutils
gamma =1
def gammaCorrection(img_original):
    ## [changing-contrast-brightness-gamma-correction]
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

    res = cv2.LUT(img_original, lookUpTable)

    return res
def basicLinearTransform(img_original,alpha,beta):
    res = cv2.convertScaleAbs(img_original, alpha=alpha, beta=beta)
    return res

def cartoonize(img_rgb):
    numDownSamples = 2       # number of downscaling steps
    numBilateralFilters = 7  # number of bilateral filtering steps

    # -- STEP 1 --
    # downsample image using Gaussian pyramid
    img_color = img_rgb
    for _ in range(0,numDownSamples):
        img_color = cv2.pyrDown(img_color)

    # repeatedly apply small bilateral filter instead of applying
    # one large filter
    for _ in range(0,numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color,9, 9, 9)

    # upsample image to original size
    for _ in range(0,numDownSamples):
        img_color = cv2.pyrUp(img_color)

    # make sure resulting image has the same dims as original
    #img_color = cv2.resize(img_color, img_rgb.shape[:2])
    img_color = imutils.resize(img_color,width = img_rgb.shape[1],height = img_rgb.shape[0])

    # -- STEPS 2 and 3 --
    # convert to grayscale and apply median blur
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)

    # -- STEP 4 --
    # detect and enhance edges
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 5, 5)

    # -- STEP 5 --
    # convert back to color so that it can be bit-ANDed with color image
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
    img_edge = imutils.resize(img_edge,width = img_rgb.shape[1])
    img_color = cv2.resize(img_color,(img_rgb.shape[1],img_rgb.shape[0]) )
    img_edge = gammaCorrection(img_edge)
    img_color = gammaCorrection(img_color)
    img_color = basicLinearTransform(img_color,120/100,100-100)
    print(img_color.shape)
    print(img_edge.shape)
    print(img_rgb.shape)
    cartoon = cv2.bitwise_and(img_color, img_edge)
    
    
    return cartoon

from eye_enlarger import LargeEyes
filename = 'Anna.jpg'
image = cv2.imread('inputs/'+filename)
print(image.shape)
le = LargeEyes()
image = le.enlarge(image)
image =imutils.resize(image, width=500)
cv2.imwrite('outputs/Cartoonize'+filename, cartoonize(image))
'''
cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    image =imutils.resize(image, width=500)
    c = cv2.waitKey(1)
    if c == 27:
        break
    cv2.imshow('Cartoonize', imutils.resize(cartoonize(image),width = 700))
cap.release()
cv2.destroyAllWindows()
        #cv2.imwrite('Cartoonize.jpg', cartoonize(image))
'''