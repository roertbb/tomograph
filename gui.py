from tkinter import * 
from PIL import ImageTk, Image
import cv2

class GUI(Frame):
    image_size = 300
    slider_len = 300
    init_detectors_num = 200
    init_angle_step = 1.0
    init_detectors_range = 180
    init_convolution = True

    def __init__(self, root):
        super().__init__()
        # tomograph params
        self.detectors_num = GUI.init_detectors_num
        self.angle_step = GUI.init_angle_step
        self.detectors_range = GUI.init_detectors_range
        self.convolution = GUI.init_convolution

        # gui
        self.root = root
        self.master.title("Tomograph")
        self.master.geometry("1000x600")
        self.pack(fill=BOTH, expand=1)
        
        self.label1 = Label(self, text="Original image")
        self.label1.place(x=170,y=330,anchor="center")
        self.label2 = Label(self, text="Sinogram")
        self.label2.place(x=490,y=330,anchor="center")
        self.label3 = Label(self, text="Output image")
        self.label3.place(x=810,y=330,anchor="center")

        self.display_image("./data/Shepp_logan.jpg", "original")
        self.display_image("./data/Shepp_logan.jpg", "sinogram")
        self.display_image("./data/Shepp_logan.jpg", "output")

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

        self.convolution_checkbox_variable = IntVar()
        self.convolution_checkbox_variable.set(self.convolution)
        self.convolution_checkbox = Checkbutton(self, text="Use convolution (noise reduction)", variable=self.convolution_checkbox_variable, command=self.convolution)
        self.convolution_checkbox.place(x=10,y=475)

        self.run_tomograph = Button(self, text="Run tomograph", command=lambda: print("run tomograph"))
        self.run_tomograph.place(x=20, y=500)

        self.run_tomograph_iterative = Button(self, text="Run tomograph (iteratively)", command=lambda: print("run tomograph (iteratively)"))
        self.run_tomograph_iterative.place(x=150, y=500)

        self.mainloop()


    def display_image(self, image, position):
        image = cv2.imread("./data/Shepp_logan.jpg",cv2.IMREAD_GRAYSCALE)
        max_size = 300
        size = image.shape[:2]
        if (size[0] > max_size or size[1] > max_size):
            image = cv2.resize(image,(int(max_size * size[1]/size[0]),int(max_size * size[0]/size[1])))
        img = ImageTk.PhotoImage(Image.fromarray(image))
        label = Label(self, image=img)
        label.image = img
        if position == "original":
            label.place(x=20, y=20)
        elif position == "sinogram":
            label.place(x=340, y=20)
        elif position == "output":
            label.place(x=660, y=20)

root = Tk()
my_gui = GUI(root)