from tkinter import * 
from tkinter import filedialog
import os
import tempfile
import datetime
import pydicom
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.filewriter import correct_ambiguous_vr
import numpy as np

class DicomWindow():
    def __init__(self, master, img):
        self.master = master
        self.master.title("save to dicom")
        self.master.geometry("380x200")
        self.img = img

        self.patient_id = ""
        self.patient_name = ""
        self.patient_sex = ""
        self.patient_birthdate = ""
        self.examination_date = ""
        self.comment = ""

        self.label1 = Label(self.master, text="Patient ID").place(x=10,y=10)
        self.patient_id = StringVar()
        self.patient_id_entry = Entry(self.master, textvariable=self.patient_id).place(x=200,y=10)
        
        self.label2 = Label(self.master, text="Patient name").place(x=10,y=35)
        self.patient_name = StringVar()
        self.patient_name_entry = Entry(self.master, textvariable=self.patient_name).place(x=200,y=35)

        self.label3 = Label(self.master, text="Patient sex").place(x=10,y=60)
        self.patient_sex = StringVar()
        self.patient_sex_entry = Entry(self.master, textvariable=self.patient_sex).place(x=200,y=60)

        self.label4 = Label(self.master, text="Patient birth date").place(x=10,y=85)
        self.patient_birthdate = StringVar()
        self.patient_birthdate_entry = Entry(self.master, textvariable=self.patient_birthdate).place(x=200,y=85)
        
        self.label5 = Label(self.master, text="Patient examination date").place(x=10,y=110)
        self.examination_date = StringVar()
        self.examination_date_entry = Entry(self.master, textvariable=self.examination_date).place(x=200,y=110)
        
        self.label6 = Label(self.master, text="Patient comment").place(x=10,y=135)
        self.comment = StringVar()
        self.comment_entry = Entry(self.master, textvariable=self.comment).place(x=200,y=135)
        
        self.save_button = Button(self.master, text="Save", command=lambda: self.save()).place(x=10, y=165)

    def save(self):
        image = np.array(self.img, dtype=np.uint16)
        # Create some temporary filenames
        suffix = '.dcm'
        filename_little_endian = tempfile.NamedTemporaryFile(suffix=suffix).name.replace('/tmp','')
        # Populate required values for file meta information
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
        file_meta.MediaStorageSOPInstanceUID = "1.2.3"
        file_meta.ImplementationClassUID = "1.2.3.4"
        # Create the FileDataset instance (initially no data elements, but file_meta
        # supplied)
        ds = FileDataset(filename_little_endian, {},
                        file_meta=file_meta, preamble=b"\0" * 128)

        
        ds.PatientID = self.patient_id.get()
        ds.PatientName = self.patient_name.get()
        ds.PatientSex = self.patient_sex.get()
        ds.PatientBirthDate = self.patient_birthdate.get()
        ds.StudyDate = self.examination_date.get()
        ds.ImageComments = self.comment.get()
        
        ds.PixelRepresentation = 1
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 1
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        ds.SmallestImagePixelValue = str.encode('\x00\x00')
        ds.LargestImagePixelValue = str.encode('\xff\xff')
        ds.Columns = image.shape[0]
        ds.Rows = image.shape[1]    
        dt = datetime.datetime.now()
        try:
            if image.max() <= 1:
                image *= 255
                image = image.astype("uint16")
            ds.PixelData = Image.fromarray(image).tobytes()
        except Exception:
            traceback.print_exc()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds = correct_ambiguous_vr(ds, True)
        filename = filedialog.asksaveasfilename(initialdir = "./",title = "choose save location",filetypes = (("dicom files","*.dcm"),("all files","*.*")) )
        if filename is None:
            return
        ds.save_as(filename)
        print("File saved.")

        