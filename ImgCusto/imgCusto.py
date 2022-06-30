from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from os import getcwd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from img_adapter import imgInvoker, tkImg2cvConverter
from img_proc_fns import average_filter, brightness_adj, contrast_adj, convertimage_tograyscale, cropimage_horizontally, cropimage_vertically, dilation, erosion, flipimage_Horizontally, flipimage_vertically, gaussian_filter, histogram_equ, low_passfilter, median_filter, rotation, scaling, sobel_edgedetector, translation


class MainWindow:
    def __init__(self,root:Tk):
        self.master=root

        self.originalImgPath='assets/lenna.png'
        self.oldColoredImg=None

        self.configWindow(root)
        self.buildImgSelectorFrame(root)
        self.buildImgViewerFrame(root)
        self.buildCommandsFrame(root)
        self.buildCredits(root)

    def configWindow(self,master:Tk):
        master.title("ImgCusto")
        master.geometry('820x650')
        master.geometry('+10+10')
        master.resizable(0,0)
    
    def buildImgSelectorFrame(self,master:Tk):
        self.imgSelectorFrame = LabelFrame(master=master,text='Image Selector')
        self.selectImgBtn = ttk.Button(master= self.imgSelectorFrame,text='Select',command=self.selectImg)
        self.selectImgBtn.pack(side=LEFT,pady=10,padx=(10,0))
        self.imgNameLbl = Label(master=self.imgSelectorFrame,text='lenna.png',font =("Courier", 12))
        self.imgNameLbl.pack(side=RIGHT,padx=10,pady=10)
        self.imgSelectorFrame.pack(side=TOP,padx=10,pady=10)
    
    def selectImg(self):
        fileName = askopenfilename(initialdir=getcwd()+"/assets",title='Select an Image',filetypes=(('PNG','*.png'),('JPG','*.jpg')))
        if fileName != '' :
            self.imgNameLbl.config(text=fileName.split('/').pop())
            self.originalImgPath = fileName
            self.clearImageViewerFrame()
            self.originalImg= Image.open(self.originalImgPath)
            self.viewImages([('Original',self.originalImg)])

    def buildImgViewerFrame(self,master:Tk):
        self.imgViewerFrame = LabelFrame(master=master,text='Image Viewer')
        self.originalImg= Image.open('assets/lenna.png')
        self.viewImages([('PlaceHolder',self.originalImg)])
        self.imgViewerFrame.pack(side=RIGHT,padx=10,pady=10,fill=BOTH,ipadx=10,ipady=5)
    
    def clearImageViewerFrame(self):
        for child in self.imgViewerFrame.winfo_children():
            child.pack_forget()

    def viewImages(self,imagesTuple):
        self.clearImageViewerFrame()
        self.photoList=[]
        # img = ('title',<IMG>)
        dim = 300 - len(imagesTuple)*50
        for img in imagesTuple:
            lf= LabelFrame(self.imgViewerFrame,text=img[0])
            c = Canvas(lf,width=dim,height=dim)
            self.photoList.append(ImageTk.PhotoImage(img[1].resize((dim,dim))))
            c.create_image(0,0,image=self.photoList[len(self.photoList)-1],anchor='nw')
            c.pack()
            lf.pack()
    
    
    def buildCommandsFrame(self,master:Tk):
        self.commandsFrame = LabelFrame(master=master,text='Commands')

        self.flipFrame = LabelFrame(master=self.commandsFrame,text='Flip Image')
        ttk.Button(master=self.flipFrame,width=20,text='Flip Vertical',command=self.onFlipVert).pack(padx=10,pady=5,anchor='w')
        ttk.Button(master=self.flipFrame,width=20,text='Flip Horizontal',command=self.onFlipHoriz).pack(padx=10,pady=(0,10),anchor='w')
        self.flipFrame.grid(row=0,column=0,sticky='nsew',padx=(10,5),pady=(10,5))

        self.convertFrame = LabelFrame(master=self.commandsFrame,text='Convert')
        self.colorMode=IntVar()
        self.colorMode.set(0)
        ttk.Radiobutton(self.convertFrame,text="RGB",variable=self.colorMode,value=0,command=self.onConvert2RGB).pack(anchor='w',padx=10,pady=(10,5))
        ttk.Radiobutton(self.convertFrame,text="Gray",variable=self.colorMode,value=1,command=self.onConvert2Gray).pack(anchor='w',padx=10,pady=(5,10))
        self.convertFrame.grid(row=0,column=1,sticky='nsew',padx=(5,5),pady=(10,5))

        self.morphFrame = LabelFrame(master=self.commandsFrame,text='Morphlogical')
        ttk.Button(master=self.morphFrame,width=20,text='Erosion',command=self.onErosion).pack(padx=10,pady=5,anchor='w')
        ttk.Button(master=self.morphFrame,width=20,text='Dilation',command=self.onDilation).pack(padx=10,pady=(0,10),anchor='w')
        self.morphFrame.grid(row=0,column=2,sticky='nsew',padx=(5,10),pady=(10,5))

        self.editFrame = LabelFrame(master=self.commandsFrame,text='Edit')
        ttk.Button(master=self.editFrame,width=20,text='Crop',command=self.onCrop).pack(padx=10,pady=5,anchor='w')
        ttk.Button(master=self.editFrame,width=20,text='Scale',command=self.onScale).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.editFrame,width=20,text='Rotate',command=self.onRotate).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.editFrame,width=20,text='Translate',command=self.onTranslate).pack(padx=10,pady=(0,10),anchor='w')
        self.editFrame.grid(row=1,column=0,sticky='nsew',padx=(10,5),pady=(5,10))

        self.pointFrame = LabelFrame(master=self.commandsFrame,text='Point Trans.')
        ttk.Button(master=self.pointFrame,width=20,text='Plot Histogram',command=self.onPlotHist).pack(padx=10,pady=5,anchor='w')
        ttk.Button(master=self.pointFrame,width=20,text='Contrast Adjust',command=self.onContrast).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.pointFrame,width=20,text='Brightness Adjust',command=self.onBrightness).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.pointFrame,width=20,text='Hist Equalization',command=self.onEqualized).pack(padx=10,pady=(0,10),anchor='w')
        self.pointFrame.grid(row=1,column=1,sticky='nsew',padx=(5,5),pady=(5,10))

        self.filtersFrame = LabelFrame(master=self.commandsFrame,text='Filters')
        ttk.Button(master=self.filtersFrame,width=20,text='Low pass',command=self.onLowPass).pack(padx=10,pady=5,anchor='w')
        ttk.Button(master=self.filtersFrame,width=20,text='Gaussian',command=self.onGaussian).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.filtersFrame,width=20,text='Averaging',command=self.onAveraging).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.filtersFrame,width=20,text='Sobel',command=self.onSobel).pack(padx=10,pady=(0,5),anchor='w')
        ttk.Button(master=self.filtersFrame,width=20,text='Median',command=self.onMedian).pack(padx=10,pady=(0,10),anchor='w')
        self.filtersFrame.grid(row=1,column=2,sticky='nsew',padx=(5,10),pady=(5,10))

        self.commandsFrame.pack(side=LEFT,padx=10,pady=10,anchor='nw')

    def onFlipVert(self):
        self.resultImg =imgInvoker(self.originalImg,flipimage_vertically)
        self.viewImages([('Original',self.originalImg),('Vertically Flipped',self.resultImg)])

    def onFlipHoriz(self):   
        self.resultImg =imgInvoker(self.originalImg,flipimage_Horizontally)
        self.viewImages([('Original',self.originalImg),('Horizontally Flipped',self.resultImg)])

    def onConvert2RGB(self):
        self.originalImg =Image.open(self.originalImgPath)
        self.viewImages([('RGB Image',self.originalImg)])
    
    def onConvert2Gray(self):
        self.originalImg =imgInvoker(self.originalImg,convertimage_tograyscale)
        self.viewImages([('Grayscale Image',self.originalImg)])
    
    def onCrop(self):
        self.result =imgInvoker(self.originalImg,cropimage_vertically)
        self.result =imgInvoker(self.result,cropimage_horizontally)
        self.viewImages([('Original Image',self.originalImg),('Cropped Image',self.result)])

    def onScale(self):
        self.result =imgInvoker(self.originalImg,scaling)
        self.viewImages([('Original Image',self.originalImg),('Scaled Image',self.result)])
    
    def onRotate(self):
        self.result =imgInvoker(self.originalImg,rotation)
        self.viewImages([('Original Image',self.originalImg),('Rotated Image',self.result)])

    def onTranslate(self):
        self.result =imgInvoker(self.originalImg,translation)
        self.viewImages([('Original Image',self.originalImg),('Translated Image',self.result)])
    
    def onPlotHist(self):
        fig = Figure(figsize=(5,4), dpi=50)
        a = fig.add_subplot(111)
        img=tkImg2cvConverter(self.originalImg)
        a.hist(img.flatten(),256,[0,256], color = 'b')
        self.viewImages([('Original',self.originalImg)])
        self.canvas = FigureCanvasTkAgg(fig, master=self.imgViewerFrame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def onContrast(self):
        self.result =imgInvoker(self.originalImg,contrast_adj)
        self.viewImages([('Original Image',self.originalImg),('Contrast Adjusted',self.result)])

    def onBrightness(self):
        self.result =imgInvoker(self.originalImg,brightness_adj)
        self.viewImages([('Original Image',self.originalImg),('Brightness Adjusted',self.result)])

    def onEqualized(self):
        self.result =imgInvoker(self.originalImg,histogram_equ)
        self.viewImages([('Original Image',self.originalImg),('Hist Equalized',self.result)])

    def onErosion(self):
        self.result =imgInvoker(self.originalImg,erosion)
        self.viewImages([('Original Image',self.originalImg),('Erosion Image',self.result)])

    def onDilation(self):
        self.result =imgInvoker(self.originalImg,dilation)
        self.viewImages([('Original Image',self.originalImg),('Dilation Image',self.result)])

    def onLowPass(self):
        self.result =imgInvoker(self.originalImg,low_passfilter)
        self.viewImages([('Original Image',self.originalImg),('Low Pass',self.result)])
    
    def onGaussian(self):
        self.result =imgInvoker(self.originalImg,gaussian_filter)
        self.viewImages([('Original Image',self.originalImg),('Gaussian',self.result)])

    def onAveraging(self):
        self.result =imgInvoker(self.originalImg,average_filter)
        self.viewImages([('Original Image',self.originalImg),('Averaging',self.result)])
    
    def onSobel(self):
        self.result =imgInvoker(self.originalImg,sobel_edgedetector)
        self.viewImages([('Original Image',self.originalImg),('Sobel',self.result)])

    def onMedian(self):
        self.result =imgInvoker(self.originalImg,median_filter)
        self.viewImages([('Original Image',self.originalImg),('Median',self.result)])

    def buildCredits(self,master:Tk):
        self.creditsFrame =LabelFrame(master=master,text='Credits')
        fnt = ("Courier", 18)
        Label(master=self.creditsFrame,text='   Mahmoud Nasser Mansour',font=fnt).pack(anchor='w')
        Label(master=self.creditsFrame,text='   Mohamed Reda Elshiekh',font=fnt).pack(anchor='w')
        self.creditsFrame.pack(side=BOTTOM,padx=10,pady=10,fill=X,before=self.commandsFrame)

if __name__ == '__main__' :
    root = Tk()
    MainWindow(root)
    root.mainloop()