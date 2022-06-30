from operator import add
from tkinter import*
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import filedialog
import math
from PIL.ImageEnhance import Brightness
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.getlimits import _fr1
from numpy.lib.type_check import imag
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
window = Tk() 
window.geometry('1250x1000+150+0')
window.title('Pattern Recognition and Image Processing')

# Right Large Frame 

fr=ttk.LabelFrame(window,height=1400,width=500,text='Screen',relief= SUNKEN )
fr.grid(row=0 , column=1, rowspan=3  )

f_Original=ttk.LabelFrame(fr,height=200,width=300,text='Original Image ' ,relief= SUNKEN )
f_Original.grid(row = 0,  column=0 )
Original_Image=Label(f_Original,text='')


f_noise=ttk.LabelFrame(fr,height=200,width=300,text=' After noise Adding' ,relief= SUNKEN )
f_noise.grid(row = 1,  column=0 )
after_noise=Label(f_noise,text='')

f_result=ttk.LabelFrame(fr,height=200,width=300,text='Result' ,relief= SUNKEN )
f_result.grid(row = 2,  column=0 )
result_=Label(f_result,text='')



def open():
    
    global filename, my_image
    filename =filedialog.askopenfilename(initialdir="/",title='select a photo',filetypes=(('jpg files','*.jpg'),("all files",'*.*')))
    #my_image=ImageTk.PhotoImage(Image.open(filename))

def convert(value,label):
    global image
    image=cv.imread(filename)
    image=cv.resize(image,(500,500))

    if value=='default':
     #   image=Image.fromarray(image)
      #  image=ImageTk.PhotoImage(image)
       # label.configure(image=image)
       # label.image=image
       # label.pack()
       cv.imshow("image ",image)
       cv.waitKey(0)
    if value=='gray':
        
        gray_image=cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image=gray_image
        #gray_image=Image.fromarray(gray_image)
        #gray_image=ImageTk.PhotoImage(gray_image)
        #label.configure(image=gray_image)
        #label.image=gray_image
        #label.pack()
        cv.imshow("image ",image)
        cv.waitKey(0)
   
def add_noise(noise_type):
        # error here
        global image
        
        if noise_type == "gauss":

           
            row,col,ch= image.shape
            mean = 0
            var = 0.01
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            image = noisy
            cv.imshow("image ",image)
            cv.waitKey(0)
            
        elif noise_type == "s&p":
            row, col, ch =image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
            out[tuple(coords)] = 255
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[tuple(coords)] = 0
            image = out
            cv.imshow("image ",image)
            cv.waitKey(0)
            
        # error here
        elif noise_type == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            image = noisy
            cv.imshow("image ",image)
            cv.waitKey(0)
            

   # Point Transformation
def bc_adj(  ):
    global image
    alpha=1.0
    beta=0
    # brightness and contrast adjustment
    new_img = np.zeros(image.shape,image.dtype)
    new_img[:,:,:] = np.clip(alpha * image[:,:,:] + beta, 0, 255)
    image = new_img
    cv.imshow("image ",image)
    cv.waitKey(0)

#contrast adjustment
def contrast_adj ():
    global image
    alpha =2
    beta = 50
    image_cv =cv.addWeighted(image,alpha,np.zeros(image.shape,image.dtype),0,beta)
    cv.imshow('Image',image)
    cv.waitKey(0)

    #Histogram graph
def histogram_graph ():
    global image
    plt.hist(image.ravel(),255,[0,255])
    plt.show()
    
#histogram equlization
def histogram_equ():
    global image
    src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(src)
    cv.imshow('Source image', src)
    cv.imshow('Equalized Image', dst)
    cv.waitKey(0)

#low pass filter
def lowpassfilter():
    global image
    image = cv.blur(image,(5,5),0)
    cv.imshow("image",image)
    cv.waitKey(0)

def highpassfilter():
    global image
    image = image- cv.GaussianBlur(image, (0,0), 3) + 127
    cv.imshow("image",image)
    cv.waitKey(0)

def medianfilter():
    global image
    image = cv.medianBlur(image,5)
    cv.imshow("image",image)
    cv.waitKey(0)

def averagingfilter():
    global image
    image= cv.boxFilter(image, -1, (10, 10), normalize=True) 
    cv.imshow("image",image)
    cv.waitKey

def apply_filter(val):
    global image
    if val == 0 :
        image = cv.Laplacian(image,cv.CV_64F)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 1:
        image = cv.GaussianBlur(image,(5,5),cv.BORDER_DEFAULT)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 2:
        image = cv.Sobel(image,cv.CV_64F,1,0,ksize=5)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 3:
        image = cv.Sobel(image,cv.CV_64F,0,1,ksize=5)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 4:
        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        image = cv.filter2D(image, -1, kernelx)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 5 :
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        image = cv.filter2D(image, -1, kernely)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 6 :
        blur = cv.GaussianBlur(image,(3,3),0)
        laplacian = cv.Laplacian(blur,cv.CV_64F)
        image = laplacian/laplacian.max()
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 7 :
        image = cv.Canny(image,100,200)
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val == 8 :
        #pip install opencv-contrib-python
        image= cv.ximgproc.thinning(cv.cvtColor(image, cv.COLOR_RGB2GRAY))
        cv.imshow("image",image)
        cv.waitKey(0)
        return
    elif val== 9:
        image= skeletiozation_img(image)
        cv.imshow("image",image)
        cv.waitKey(0)
        return

def skeletiozation_img(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv.threshold(img,127,255,0)
    element = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv.erode(img,element)
        temp = cv.dilate(eroded,element)
        temp = cv.subtract(img,temp)
        skel = cv.bitwise_or(skel,temp)
        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def LineDetection():
    global image
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    lines = cv.HoughLines(edges,1,np.pi/180,200)
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(image,(x1,y1),(x2,y2),(0,0,255),2)
    cv.imshow("image",image)
    cv.waitKey(0)
    return



def save_image ():
    cv.imwrite("/image_result.jpg",image)
    return

#Left Large frame
fl=ttk.LabelFrame(window,height=1400,width=700,text='Image Options',relief= SUNKEN )
fl.grid(row=0 , column=0 ,rowspan = 4 , columnspan=1 )


#load image frame 
f1=ttk.LabelFrame(fl,height=150,width=150,text='Load Image',relief= SUNKEN )
f1.grid(row=0 , column=0 )

b1 = ttk.Button( f1, text='Open ' , cursor='hand2',command=open)
b1.pack()
f1.config(padding=(25,25))

#Convert image frame  
butt_value=StringVar()
butt_value.set('default')
f2=ttk.LabelFrame(fl,height=150,width=150,text='Convert ',relief= RIDGE)
f2.grid(row=0 , column=1 )

rad1 = ttk.Radiobutton(f2 , text='Default color',variable=butt_value,value='default',command=lambda: convert(butt_value.get(),Original_Image) )
rad1.pack(anchor = W )

rad2 = ttk.Radiobutton(f2 , text='Gray color',variable=butt_value,value='gray',command=lambda:convert(butt_value.get(),Original_Image)) 
rad2.pack(anchor = W )

#Add noise frame 


f3=ttk.LabelFrame(fl,height=150,width=150,text='Add Noise',relief= RIDGE )
f3.grid(row=0 , column=2  )

noise_value=StringVar()
noise_value.set('s&p')
rad1 = ttk.Radiobutton(f3 , text=' salt & peper noise',variable=noise_value,value='s&p',command=lambda:add_noise(noise_value.get()) )
rad1.pack(anchor = W )

rad2 = ttk.Radiobutton(f3 , text='Gaussian noise ',variable=noise_value,value='gauss',command=lambda: add_noise(noise_value.get()))
rad2.pack(anchor = W )

rad3 = ttk.Radiobutton(f3 , text=' Poisson noise',variable=noise_value,value='poisson',command=lambda: add_noise(noise_value.get()) )
rad3.pack(anchor = W )

# Point Transform Frame 

f4=ttk.LabelFrame(fl,height=150,width=500,text='Point Transform Op\'s  ',relief= SUNKEN )
f4.grid ( row=1, columnspan=3  )


b1 = ttk.Button( f4, text='Brightness Adjustment ' , cursor='hand2', width=20,command=bc_adj)
b1.grid(row = 0 , column = 0)


b2 = ttk.Button( f4, text='Contrast Adjustment ' , cursor='hand2', width=20,command=contrast_adj)
b2.grid(row = 1 , column = 1)


b3 = ttk.Button( f4, text='Histogram  ' , cursor='hand2' , width=20,command=histogram_graph)
b3.grid(row = 2 , column = 2)


b4 = ttk.Button( f4, text='Histogram Equalization ' , cursor='hand2' , width=20,command=histogram_equ)
b4.grid(row = 3 , column = 3)


# Local Transform Frame 

f5=ttk.LabelFrame(fl,height=150,width=500,text='Local Transform Op\'s  ',relief= SUNKEN )
f5.grid ( row=2 ,rowspan = 4 , columnspan=2 )


b1 = ttk.Button( f5, text="Low-pass filter" , cursor='hand2', width=20,command=lowpassfilter)
b1.grid(row = 0 , column = 0)


b2 = ttk.Button( f5, text="High-pass filter" , cursor='hand2', width=20,command=highpassfilter)
b2.grid(row = 1 , column = 0)


b3 = ttk.Button( f5, text=" Median filter" , cursor='hand2', width=20,command=medianfilter)
b3.grid(row = 2 , column = 0)


b4 = ttk.Button( f5, text="Averaging filter" , cursor='hand2' , width=20,command=averagingfilter)
b4.grid(row = 3 , column = 0)

# Edge Detection Frame inside Local Transform Frame 
var =IntVar()

f5_1=ttk.LabelFrame(f5,height=150,width=300,text='Edge Detection Filters ',relief= SUNKEN )
f5_1.grid ( row = 0 , column=1 , rowspan = 4 , columnspan=4)

rad1 = ttk.Radiobutton(f5_1 , text='Laplacian ' ,variable= var,value=0,command=lambda :apply_filter(var.get()))
rad1.grid( row = 0 , column = 0  )

rad2 = ttk.Radiobutton(f5_1 , text='Gaussian  ',variable= var,value=1,command=lambda :apply_filter(var.get()) )
rad2.grid( row = 0 , column = 1  )

rad3 = ttk.Radiobutton(f5_1 , text='Vert.Sobel  ',variable= var,value=2,command=lambda :apply_filter(var.get()) )
rad3.grid( row = 0 , column = 2  )

rad4 = ttk.Radiobutton(f5_1, text='Horiz.Sobel ' ,variable= var,value=3,command=lambda :apply_filter(var.get()))
rad4.grid( row = 0 , column = 3  )

rad5 = ttk.Radiobutton(f5_1, text='Vert.prewitt ' ,variable= var,value=4,command=lambda :apply_filter(var.get()))
rad5.grid( row = 1 , column = 0  )

rad6 = ttk.Radiobutton(f5_1, text='Horiz.prewitt ' ,variable= var,value=5,command=lambda :apply_filter(var.get()))
rad6.grid( row = 1 , column = 1 )

rad7 = ttk.Radiobutton(f5_1, text='Lap of Gau (LOG) ' ,variable= var,value=6,command=lambda :apply_filter(var.get()))
rad7.grid( row = 1 , column = 2 )

rad8 = ttk.Radiobutton(f5_1, text='Canny  method' ,variable= var,value=7,command=lambda :apply_filter(var.get()))
rad8.grid( row = 1 , column = 3 )

rad9 = ttk.Radiobutton(f5_1 , text='Zero Cross' )
rad9.grid( row = 2 , column = 0 )

rad10 = ttk.Radiobutton( f5_1 , text='Thicken ' )
rad10.grid( row = 2 , column = 1 )

rad11 = ttk.Radiobutton( f5_1 , text='Skeleton ' ,variable= var,value=9,command=lambda :apply_filter(var.get()))
rad11.grid( row = 2 , column = 2 )

rad12 = ttk.Radiobutton( f5_1 , text='Thinning' ,variable= var,value=8,command=lambda :apply_filter(var.get()))
rad12.grid( row = 2 , column = 3  )


#Global Transform Frame 

f6=ttk.LabelFrame(fl,height=150,width=300,text='Global Transform Op\'s  ',relief= SUNKEN )
f6.grid(row = 6 , column=0)


b1 = ttk.Button( f6, text='Line Detection Using Hough Transform ' , cursor='hand2', width=40,command=LineDetection )
b1.pack()
f6.config(padding=(25,25))

b2 = ttk.Button( f6, text='Circle Detection Using Hough Transform ' , cursor='hand2' , width=40)
b2.pack(pady=15)
f6.config(padding=(25,25))


#Morphological Frame

f7=ttk.LabelFrame(fl,height=150,width=300,text='Morphological Op\'s ' ,relief= SUNKEN )
f7.grid(row = 6,  column=1 )


b1 = ttk.Button( f7, text='Dilation' , cursor='hand2', width=15)
b1.grid(row = 0 , column = 0)


b2 = ttk.Button( f7, text='Erosion' , cursor='hand2', width=15)
b2.grid(row = 1 , column = 0)


b3 = ttk.Button( f7, text='Close' , cursor='hand2' , width=15)
b3.grid(row = 2 , column = 0)


b4 = ttk.Button( f7, text='Open' , cursor='hand2' , width=15)
b4.grid(row = 3 , column = 0)

l = Label(f7,text='Choose type of Kernal')
l.grid(row=0,column=1)
c = ttk.Combobox(f7)
c.grid(row=1, column=1)
c.config(value=('1','2' ,'3' ,'4' ,'5' ,'6' ,'7' ,'8' ,'9' ,'10' ,'11' ,'12') )
c.set('Arbitrary')

# Save Button 

b = ttk.Button( fl, text='Save Result Image' , cursor='hand2', width=20,command=save_image)
b.grid(row=7 , column=0,padx=20 , pady=50)


#Exit Button 


b = ttk.Button( fl, text='Exit' , cursor='hand2', width=10, command=window.destroy)
b.grid(row=7 , column=1,padx=20 , pady=50)




window.mainloop()