from pyclbr import Function
import cv2
from PIL import Image
def cv2jpgConverter(img):
    result_path='assets/result.jpg'
    cv2.imwrite(result_path,img)
    return Image.open(result_path)

def tkImg2cvConverter(img:Image):
    return cv2.imread(img.filename)

def imgInvoker(img:Image,func:Function):
    img = tkImg2cvConverter(img)
    res = func(img)
    return cv2jpgConverter(res)