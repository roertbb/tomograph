import cv2
from matplotlib import pyplot as plt
import math
import numpy as np
import time
import multiprocessing as mp

def plot_image(img):
    plt.imshow(img,cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def load(img_name):
    max_size = 400
    image = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
    size = image.shape[:2]
    if (size[0] > max_size or size[1] > max_size):
        image = cv2.resize(image,(int(max_size * size[1]/size[0]),int(max_size * size[0]/size[1])))
    return image
    # old_size = image.shape[:2] 
    # img = cv2.resize(image, (0,0), fx=0.5, fy=0.5) 
    # delta_x = int(old_size[1]/4)
    # delta_y = int(old_size[0]/4)
    # color = [0, 0, 0]
    # resized_image = cv2.copyMakeBorder(img, delta_y, delta_y, delta_x, delta_x, cv2.BORDER_CONSTANT,value=color)
    # return resized_image

# get pixels position within line based on Bresenhama algorithm
def gen_line(x1,y1,x2,y2):
    line = []
    
    xi = 1 if x1<x2 else -1
    yi = 1 if y1<y2 else -1
    dx = x2-x1 if x1<x2 else x1-x2
    dy = y2-y1 if y1<y2 else y1-y2
    x = x1
    y = y1
    
    line.append((x,y))
    
    if dx > dy:
        ai = (dy-dx)*2
        bi = dy*2
        d = bi-dx
        while (x != x2):
            if (d >= 0): 
                x += xi;
                y += yi;
                d += ai;
            else:
                d += bi;
                x += xi;
            line.append((x,y))
    else:
        ai = (dx-dy)*2
        bi = dx*2
        d = bi-dy
        while (y != y2):
            if (d >= 0): 
                x += xi;
                y += yi;
                d += ai;
            else:
                d += bi;
                y += yi;
            line.append((x,y))
    return line

# generate emiters position according to angle
def gen_emiter_pos(size, delta_alpha):
    w,h = size
    center_x = w/2
    center_y = h/2
    r = min(w-5,h-5)
    samples = int(360/delta_alpha)
    x = [round(math.cos(math.radians(delta_alpha*i))*r + center_x) for i in range(samples)]
    y = [round(math.sin(math.radians(delta_alpha*i))*r + center_y) for i in range(samples)]
    return list(zip(x,y))

# generate all detectors position according to angle
def get_detectors_pos(size, delta_alpha, n, l):
    detectors = []
    w,h = size
    center_x = w/2
    center_y = h/2
    r = min(w-5,h-5)
    samples = int(360/delta_alpha)
    tpd = l/n # angle translation per detector
    for i in range(n):
        translation = 180 - (n/2 * tpd) + (i * tpd) + 1/2 * tpd
        x = [round(math.cos(math.radians(delta_alpha*i + translation))*r + center_x) for i in range(samples)]
        y = [round(math.sin(math.radians(delta_alpha*i + translation))*r + center_y) for i in range(samples)]
        detectors.append(list(zip(x,y)))
    return detectors

def normalize(img):
    cp = img[:]
    maximum = np.amax(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            cp[i][j] = img[i][j]/maximum * 256
    return cp


def gen_sinogram(img, emiter_pos, detectors_pos, size):
    sinogram = np.zeros(shape=(n,len(emiter_pos)))
    for it in range(len(emiter_pos)): #iterations
        for detector_id in range(len(detectors_pos)):
            emit_x, emit_y = emiter_pos[it]
            det_x, det_y = detectors_pos[detector_id][it]
            line = gen_line(emit_x, emit_y, det_x, det_y)
            value = 0
            counter = 0
            for (x,y) in line:
                if x > 0 and y > 0 and x < size[0] and y < size[1]:
                    value += img[x][y]
                    counter += 1
            sinogram[detector_id][it] = 0 if counter == 0 else int(value/counter)
    return sinogram

def normalize_image(size, image, counter):
    x,y = size
    cp = image[:]
    maximum = 0
    for i in range(x):
        for j in range(y):
            if int(counter[i][j]) > 0 and image[i][j] > 0:
                maximum = image[i][j]
    for i in range(x):
        for j in range(y):
            if int(counter[i][j]) > 0 and image[i][j] > 0:
                cp[i][j] = image[i][j]/maximum * 255
    return cp
    

def gen_image(size, sinogram, emiter_pos, detectors_pos):
    x, y = size
    image = np.zeros(shape=size)
    counter = np.zeros(shape=size)    
    for it in range(len(emiter_pos)): #iterations
        for detector_id in range(len(detectors_pos)):
            emit_x, emit_y = emiter_pos[it]
            det_x, det_y = detectors_pos[detector_id][it]
            line = gen_line(emit_x, emit_y, det_x, det_y)
            for (x,y) in line:
                if x > 0 and y > 0 and x < size[0] and y < size[1]:
                    image[x][y] += sinogram[detector_id][it]
                    counter[x][y] += 1
    for i in range(x):
        for j in range(y):
            image[i][j] = image[i][j]/counter[i][j]

    normalized_image = normalize_image(size, image, counter)
    return normalized_image

def lino_convolution(sinogram,i,j,mask,padding):
    s = 0
    for x in range(len(mask)):
        s += sinogram[i-padding+x][j]
    return s/len(mask)

def sinogram_convolution(sinogram):
    x,y = sinogram.shape
    mask = [-1,-2,7,-2,-1]
    cp = sinogram[:]
    padding = int(len(mask)/2)
    for i in range(padding, x-padding):
        for j in range(y):
            cp[i][j] = lino_convolution(sinogram,i,j,mask,padding)
    return cp

if __name__ == "__main__":    
    delta_alpha = 1 # detector/emiter step
    n = 400 # number of detectors
    l = 230 # detector/emiter span

    start = time.time()

    img = load('./data/Shepp_logan.jpg')
    size = img.shape[:2]
    start = time.time()

    emiter_pos = gen_emiter_pos(size, delta_alpha)
    detectors_pos = get_detectors_pos(size, delta_alpha, n, l)
    sinogram = gen_sinogram(img, emiter_pos, detectors_pos, size)
    print(time.time() - start)

    normalize_sin = sinogram_convolution(sinogram)
    normalize_sin = normalize(normalize_sin)
    # plot_image(normalize_sin)
    image = gen_image(size, normalize_sin, emiter_pos, detectors_pos)
    normalized_image = normalize(image)

    print(time.time()-start)
    
    plot_image(normalized_image)