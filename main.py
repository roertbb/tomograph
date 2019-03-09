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

if __name__ == "__main__":    
    delta_alpha = 5 # detector/emiter step
    n = 30 # number of detectors
    l = 60 # detector/emiter span

    img = load('./data/03.png')
    size = img.shape[:2]
    emiter_pos = gen_emiter_pos(size, delta_alpha)
    detectors_pos = get_detectors_pos(size, delta_alpha, n, l)
    sinogram = gen_sinogram(img, emiter_pos, detectors_pos)
    normalize_sin = normalize(sinogram)
    plot_image(normalize_sin)