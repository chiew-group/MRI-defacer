from tkinter import *
from tkinter import filedialog

def select_data():
    window = Tk()
    window.withdraw()
    filename = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("All files", "*.*"),))
    return filename
                                                                                                 
