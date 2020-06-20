import tkinter as tk
import os


def write_slogan():
        os.system('python test.py')

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()
root.title('Run Detect Face Play Soccer')

back = tk.Frame( width=500, height=500)
back.pack()

button = tk.Button(frame,text="QUIT",fg="red",command=quit)
button.pack(side=tk.BOTTOM)


slogan = tk.Button(frame,text="Run",command=write_slogan)
slogan.pack(side=tk.BOTTOM)

root.mainloop()