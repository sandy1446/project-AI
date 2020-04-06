from tkinter import *
import time

my_window = Tk()
my_canvas = Canvas(my_window,width=400,height=400,background='white')
my_canvas.grid(row=0,column=0)
i=0
j=0
my_canvas.create_line(i,j,400-i,400-j,fill='red')

 
my_window.mainloop()
