import cv2
from matplotlib import pyplot as plt
import math
import numpy as np

def plot_image(img):
    plt.imshow(img,cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def load(img_name):
    return cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)

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
    r = min(w/2-5,h/2-5)
    x = [round(math.cos(math.radians(delta_alpha*i))*r + center_x) for i in range(int(360/delta_alpha))]
    y = [round(math.sin(math.radians(delta_alpha*i))*r + center_y) for i in range(int(360/delta_alpha))]
    return list(zip(x,y))

# generate all detectors position according to angle
def get_detectors_pos(size, delta_alpha, n, l):
    detectors = []
    w,h = size
    center_x = w/2
    center_y = h/2
    r = min(w/2-5,h/2-5)
    tpd = l/n # angle translation per detector
    for i in range(n):
        translation = 180 - (n/2 * tpd) + (i * tpd) + 1/2 * tpd
        x = [round(math.cos(math.radians(delta_alpha*i + translation))*r + center_x) for i in range(int(360/delta_alpha))]
        y = [round(math.sin(math.radians(delta_alpha*i + translation))*r + center_y) for i in range(int(360/delta_alpha))]
        detectors.append(list(zip(x,y)))
    return detectors

# def generate_all_lines(emiter_pos, detectors_pos):
#     lines = []
#     for i in range(len(emiter_pos)):
#         lines_per_angle = []
#         for detector_pos in detectors_pos[i]:
#             emiter_x, emiter_y = emiter_pos[i]
#             detector_x, detector_y = detector_pos
#             line = gen_line(emiter_x, emiter_y, detector_x, detector_y)
#             lines_per_angle.append(line)
#         lines.append(lines_per_angle)
#     return lines

def normalize(img):
    cp = img[:]
    maximum = np.amax(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            cp[i][j] = img[i][j]/maximum * 266
    return cp


def gen_sinogram(img, emiter_pos, detectors_pos):
    sinogram = np.zeros(shape=(n,len(emiter_pos)))
    for it in range(len(emiter_pos)): #iterations
        for detector_id in range(len(detectors_pos)):
            emit_x, emit_y = emiter_pos[it]
            det_x, det_y = detectors_pos[detector_id][it]
            line = gen_line(emit_x, emit_y, det_x, det_y)
            values = [img[x][y] for (x,y) in line]
            sinogram[detector_id][it] = sum([x/len(line) for x in values])
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
            # print(sinogram[detector_id][it])
            for (x,y) in line:
                image[x][y] = image[x][y] + sinogram[detector_id][it]
                counter[x][y] = counter[x][y] + 1
    for i in range(x):
        for j in range(y):
            image[x][y] = image[x][y]/counter[x][y]

    normalized_image = normalize_image(size, image, counter)
    # return image
    return normalized_image

def calc_mask_value(img, mask, i, j, mask_padding):
    y = len(mask)
    x = len(mask[0])
    value = 0
    for y in range(i-mask_padding,i+mask_padding+1):
        for x in range(j-mask_padding,j+mask_padding+1):
            value += img[y][x]/(len(mask)**2)
    return value

def apply_mask(img, size, mask):
    cp = img[:]
    w,h = size
    mask_padding = int(len(mask)/2)
    for i in range(mask_padding,w-mask_padding):
        for j in range(mask_padding,h-mask_padding):
            cp[i][j] = calc_mask_value(img, mask, i, j, mask_padding)
    return normalize(cp)


if __name__ == "__main__":    
    delta_alpha = 5 # detector/emiter step
    n = 50 # number of detectors
    l = 90 # detector/emiter span

    img = load('./data/Shepp_logan.jpg')
    size = img.shape[:2]
    emiter_pos = gen_emiter_pos(size, delta_alpha)
    detectors_pos = get_detectors_pos(size, delta_alpha, n, l)
    sinogram = gen_sinogram(img, emiter_pos, detectors_pos)
    normalize_sin = normalize(sinogram)
    # plot_image(normalize_sin)
    image = gen_image(size, normalize_sin, emiter_pos, detectors_pos)
    mask = [
        [0,0,0,-1,0,0,0],
        [0,-1,-1-1,-1,-1,0],
        [0,-1,-1,3,-1,-1,0],
        [0,-1,3,3,3,-1,0],
        [0,-1,-1,3,-1,-1,0],
        [0,-1,-1-1,-1,-1,0],
        [0,0,0,-1,0,0,0],
        ]
    masked_img = apply_mask(image, size, mask)
    normalized_image = normalize(masked_img)
    plot_image(normalized_image)