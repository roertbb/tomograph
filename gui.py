from tkinter import * 
from PIL import ImageTk, Image
import cv2
import numpy as np
from main import gen_emiter_pos, get_detectors_pos, gen_sinogram, gen_image, normalize_image_iterative

class GUI(Frame):
    image_size = 300
    slider_len = 300
    init_detectors_num = 200
    init_angle_step = 1.0
    init_detectors_range = 180
    init_convolution = True
    img_max_size = 300

    def __init__(self, root):
        super().__init__()
        # tomograph params
        self.detectors_num = GUI.init_detectors_num
        self.angle_step = GUI.init_angle_step
        self.detectors_range = GUI.init_detectors_range
        self.convolution = GUI.init_convolution
        self.original_image = np.ones([GUI.image_size,GUI.image_size])
        self.sinogram = np.ones([GUI.image_size,GUI.image_size])
        self.generated_image = np.ones([GUI.image_size,GUI.image_size])

        self.original_image_label = None
        self.sinogram_label = None
        self.generated_image_label = None

        # gui
        self.root = root
        self.master.title("Tomograph")
        self.master.geometry("980x600")
        self.pack(fill=BOTH, expand=1)
        
        self.label1 = Label(self, text="Original image")
        self.label1.place(x=170,y=330,anchor="center")
        self.label2 = Label(self, text="Sinogram")
        self.label2.place(x=490,y=330,anchor="center")
        self.label3 = Label(self, text="Output image")
        self.label3.place(x=810,y=330,anchor="center")

        self.display_image("original")
        self.display_image("sinogram")
        self.display_image("generated")

        self.params = Label(self, text="Tomograph parameters:")
        self.params.place(x=20,y=360)

        self.detectors_num_label = Label(self, text="n - Number fo detectors:")
        self.detectors_num_label.place(x=20,y=390)
        self.detectors_num_string = StringVar()
        self.detectors_num_input = Entry(self, textvariable=self.detectors_num_string)
        self.detectors_num_string.set(GUI.init_detectors_num)
        self.detectors_num_input.place(x=220,y=388)

        self.angle_step_label = Label(self, text="Δα - Angle step (in degrees):")
        self.angle_step_label.place(x=20,y=420)
        self.angle_step_string = StringVar()
        self.angle_step_input = Entry(self, textvariable=self.angle_step_string)
        self.angle_step_string.set(GUI.init_angle_step)
        self.angle_step_input.place(x=220,y=418)

        self.detectors_range_label = Label(self, text="l - detectors range (in degrees):")
        self.detectors_range_label.place(x=20,y=450)
        self.detectors_range_string = StringVar()
        self.detectors_range_input = Entry(self, textvariable=self.detectors_range_string)
        self.detectors_range_string.set(GUI.init_detectors_range)
        self.detectors_range_input.place(x=220,y=448)

        self.convolution_checkbox_variable = IntVar()
        self.convolution_checkbox_variable.set(self.convolution)
        self.convolution_checkbox = Checkbutton(self, text="Use convolution (noise reduction)", variable=self.convolution_checkbox_variable, command=self.convolution)
        self.convolution_checkbox.place(x=10,y=475)

        self.run_tomograph = Button(self, text="Run tomograph", command=lambda: self.process())
        self.run_tomograph.place(x=20, y=500)

        self.run_tomograph_iterative = Button(self, text="Run tomograph (iteratively)", command=lambda: self.process(iteratively=True))
        self.run_tomograph_iterative.place(x=150, y=500)

        self.status = Label(self, text="Status: -")
        self.status.place(x=20,y=535)
        
        self.load_image("./data/Shepp_logan.jpg")

        self.mainloop()

    def display_image(self, position):
        if position == "original":
            img = ImageTk.PhotoImage(Image.fromarray(self.original_image))
            self.original_image_label = Label(self, image=img)
            self.original_image_label.image = img
            self.original_image_label.place(x=20+GUI.image_size/2, y=20+GUI.image_size/2,anchor="center")
        elif position == "sinogram":
            sinogram_to_display = self.resize_image_to_display(self.sinogram[:])
            img = ImageTk.PhotoImage(Image.fromarray(sinogram_to_display))
            self.sinogram_label = Label(self, image=img)
            self.sinogram_label.image = img
            self.sinogram_label.place(x=340+GUI.image_size/2, y=20+GUI.image_size/2,anchor="center")
        elif position == "generated":
            img = ImageTk.PhotoImage(Image.fromarray(self.generated_image))
            self.generated_image_label = Label(self, image=img)
            self.generated_image_label.image = img
            self.generated_image_label.place(x=660+GUI.image_size/2, y=20+GUI.image_size/2,anchor="center")

    def resize_image_to_display(self,img):
        size = img.shape[:2]
        if (size[0] > GUI.img_max_size or size[1] > GUI.img_max_size):
            img = cv2.resize(img,(int(GUI.img_max_size * size[1]/max(size[0],size[1])),int(GUI.img_max_size * size[0]/max(size[0],size[1]))))
        return img
            
    def load_image(self, img_path):
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = self.resize_image_to_display(img)
        self.original_image = img
        self.display_image("original")

    def update_sinogram(self,sinogram):
        self.sinogram = sinogram
        self.display_image("sinogram")
        self.update()

    def update_image(self,size,image,counter):
        self.generated_image = normalize_image_iterative(size,image,counter)
        self.display_image("generated")
        self.update()

    def start_processing(self):
        self.status.config(text="Status: Processing")
        self.run_tomograph.config(state="disabled")
        self.run_tomograph_iterative.config(state="disabled")
        self.sinogram = np.ones([GUI.image_size,GUI.image_size])
        self.display_image("sinogram")
        self.update()

    def processing_finished(self):
        self.status.config(text="Status: Processing finished")
        self.run_tomograph.config(state="normal")
        self.run_tomograph_iterative.config(state="normal")
        
    def process(self, iteratively=False):
        self.start_processing()

        sinogram_callback = self.update_sinogram if iteratively else None
        generate_image_callback = self.update_image if iteratively else None
        
        n, l, delta_a, doConvolution = self.validate_input()
        img_size = self.original_image.shape[:2]
        emiter_pos = gen_emiter_pos(img_size, delta_a)
        detectors_pos = get_detectors_pos(img_size, delta_a, n, l)
        sinogram = gen_sinogram(self.original_image, emiter_pos, detectors_pos, img_size, doConvolution, callback=sinogram_callback)
        self.sinogram = sinogram
        self.display_image("sinogram")
        self.update()
        generated_image = gen_image(img_size, sinogram, emiter_pos, detectors_pos, callback=generate_image_callback)
        self.generated_image = generated_image
        self.display_image("generated")

        self.processing_finished()

    def validate_input(self):
        try:
            n = int(self.detectors_num_string.get())
            l = int(self.detectors_range_string.get())
            delta_a = float(self.angle_step_string.get())
            doConvolution = bool(self.convolution_checkbox_variable.get())
            return n, l, delta_a, doConvolution
        except:
            print("invalid input")

    
root = Tk()
my_gui = GUI(root)