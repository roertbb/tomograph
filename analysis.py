from main import load, gen_emiter_pos, get_detectors_pos, gen_sinogram, gen_image,apply_window
from matplotlib import pyplot as plt
import numpy as np

def mse_compare(image1, image2, size):
    return np.square(np.subtract(image1, image2)).mean()

def iteration_error():
    errors = []
    delta_alpha = 1
    n = 400 
    l = 230

    img = load('./data/Shepp_logan.jpg')
    size = img.shape[:2]

    def calculate_error(size, image, counter):
        height, width = size
        cpimg = image[:]
        for y in range(height):
            for x in range(width):
                if counter[y][x] > 0:
                    cpimg[y][x] = cpimg[y][x]/counter[y][x]
        errors.append(mse_compare(img, cpimg, size))

    emiter_pos = gen_emiter_pos(size, delta_alpha)
    detectors_pos = get_detectors_pos(size, delta_alpha, n, l)
    sinogram = gen_sinogram(img, emiter_pos, detectors_pos, size)
    sinogram = apply_window(sinogram)
    generated_image = gen_image(size, sinogram, emiter_pos, detectors_pos, callback=calculate_error)
    
    plt.xlabel('iteracja')
    plt.ylabel('średni błąd średniokwadratowy')
    plt.title('Średni błąd średniokwadratowy w funkcji iteracji')
    plt.plot(errors)
    plt.show()

def calculate_error(alphas=[1], nn=[300],ll=[180]):

    name = ""
    if len(alphas) > 1:
        name = "delta_a.txt"
    elif len(nn) > 1:
        name = "n.txt"
    elif len(ll) > 1:
        name = "l.txt"

    file = open(name,"w") 

    img = load('./data/Shepp_logan.jpg')
    size = img.shape[:2]

    for delta_alpha in alphas:
        for n in nn:
            for l in ll:
                print("starting {};{};{}".format(delta_alpha,n,l))
                emiter_pos = gen_emiter_pos(size, delta_alpha)
                detectors_pos = get_detectors_pos(size, delta_alpha, n, l)
                sinogram = gen_sinogram(img, emiter_pos, detectors_pos, size)
                sinogram = apply_window(sinogram)
                generated_image = gen_image(size, sinogram, emiter_pos, detectors_pos)
                error = mse_compare(img, generated_image, size)

                plt.imshow(generated_image,cmap='gray', vmin = 0, vmax = 255)
                plt.savefig("./img/{}-{}-{}.png".format(delta_alpha,n,l),)

                print("{};{};{};{}".format(delta_alpha,n,l,error))
                file.write("{};{};{};{}\n".format(delta_alpha,n,l,error)) 

    file.close() 

def mask_error():
    delta_alpha = 1
    n = 400 
    l = 230

    file = open("masks.txt","w") 

    img = load('./data/Shepp_logan.jpg')
    size = img.shape[:2]

    masks = [
        [1],
        [-2,5,-2],
        [-1,-2,5,-2,-1],
        [-3,7,-3],
        [-4,9,-4]
    ]
    errors=[]

    for mask in masks:
        print("starting {};".format(mask))
        emiter_pos = gen_emiter_pos(size, delta_alpha)
        detectors_pos = get_detectors_pos(size, delta_alpha, n, l)
        sinogram = gen_sinogram(img, emiter_pos, detectors_pos, size, mask=mask)
        sinogram = apply_window(sinogram)
        generated_image = gen_image(size, sinogram, emiter_pos, detectors_pos)
        error = mse_compare(img, generated_image, size)
        file.write("{};{}\n".format(mask,error)) 
        print("{};{}".format(mask,error)) 
    
    file.close()

if __name__ == "__main__":
    # mask_error()
    # iteration_error()
    calculate_error(nn=list(range(100,400,15)))
    calculate_error(ll=list(range(50,250,10)))
    calculate_error(alphas=list(np.arange(0.5,5.5,0.25)))