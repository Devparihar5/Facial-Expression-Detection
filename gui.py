from tkinter import *
import os
import subprocess


root = Tk()
root.title("Emotion Detection & Automatic Song Recomendation using Face Recognition")
root.geometry("1000x563")

C = Canvas(root,bg = 'blue', height = 250, width = 300)
#filename = PhotoImage(file = "emoji.jpg")
#background_label = Label(root,image = filename)
#background_label.okace(x = 0,y = 0, relwidth = 1, relheight = 1)

def proceed():
    root.destroy()
    root.quit()
    os.system("emotion_detection.py")
    #os.system("python part2.py")

b1 = Button(root,text = 'Quit', font = ("Times New Roman",16),command = root.quit)
b1.pack(side = BOTTOM,padx = 5,pady = 1)

b1 = Button(root,text = 'Start', font = ("Times New Roman",16),command = proceed)
b1.pack(side = BOTTOM,padx = 5,pady = 5)

root.mainloop()
