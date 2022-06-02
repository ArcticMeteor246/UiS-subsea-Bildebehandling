import math
from re import A
import cv2
from cv2 import StereoBM
from cv2 import mean
import numpy as np
import time
from sys import platform
from common import Cursor
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import statistics
from focusvindu import Ui_Auto_focus_settings
from PyQt5 import (QtGui, QtWidgets)
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, )
import sys
from PyQt5.QtWidgets import QApplication

class Camera():
    def __init__(self, id:int, width:int=2560, height:int=720, framerate:int=30 ) -> None:
        self.id = id
        self.height = height
        self.middley = int(height/2) # Center of picture y cord
        self.width = width
        self.hud = False
        tru_width = int(width/2)
        self.length = int(width/18) # Long horisontal line for pitch
        self.length2 = int(width/22) # Short horisontal line for pitch
        self.length3 = int(width/36) # Cursor length
        self.length4 = int(self.length3/4) # Cursor spacing and triangle side length
        self.center = (width/4, height/2) # Center of picture
        self.squarestart = [int(self.center[0]-self.length-self.length4-100), int(self.center[1]-self.length)]
        self.squarestop = [int(self.center[0]-self.length-self.length4-60), int(self.center[1]+self.length)]
        self.cursor = Cursor(self.length3, self.length4, self.center)
        self.left = int(width/4-self.length3/2)
        self.right = int(width/4+self.length3/2)
        self.color = (0, 255, 0)
        self.sensor = {"gyro": (0, 0, 0)}
        if platform == "linux" or platform == "linux2":
            self.feed = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        else:
            self.feed = cv2.VideoCapture(self.id)
        self.set_picture_size(self.width, self.height)
        self.feed.set(cv2.CAP_PROP_FPS, framerate)
        #self.feed.set(cv2.CAP_PROP_AUTOFOCUS, 3)
         
        #self.feed.set(cv2.CAP_PROP_CONTRAST , -20)
        #self.feed.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        #self.feed.set(cv2.CAP_PROP_EXPOSURE, 50)
        #self.feed.set(cv2.CAP_PROP_AUTO_WB, 1)
        print(self.feed.get(cv2.CAP_PROP_FPS))

    def set_picture_size(self, width:int=2560, height:int=960):
        self.feed.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.feed.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.feed.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.height = int(self.feed.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.feed.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(f'{self.width}:{self.height}')
        self.crop_width = int(self.width/2)

    def aq_image(self, double:bool=False, t_pic:bool=False):
        #ref, frame = self.feed.read()
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        #ref, frame = self.feed.read()
        ref = self.feed.grab()
        if ref:
            _, frame = self.feed.retrieve(0)
        else:
            if double:
                return False, False
            else:
                print(time.asctime())
                return False
        if frame is None:
            if double:
                return False, False
            else:
                print(time.asctime())
                return False
        #frame = cv2.rotate(frame, cv2.ROTATE_180)
        crop = frame[:self.height, :self.crop_width]
        crop2 = frame[:self.height,self.crop_width:]
        if t_pic:
            t = time.asctime()
            cv2.imwrite(f'/home/subsea/Bilete/rov/pic_left{t}.png',crop)
            cv2.imwrite(f'/home/subsea/Bilete/rov/pic_right{t}.png', crop2)
        if double:
            crop2 = frame[:self.height,self.crop_width:]
            return crop, crop2
        else:
            return crop

    ## Draws on image
    def draw_on_img(self, pic, frames):
        if isinstance(frames, list):
            if frames != []:
                for item in frames: # Draws objects on picture
                    cv2.rectangle(pic, item.rectangle[0], item.rectangle[1], item.colour, item.draw_line_width) # Draws rectablge on picture
                    pos = (item.rectangle[0][0], item.rectangle[0][1]+40) # For readability
                    cv2.putText(pic, item.name, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3) # Red text
                    cv2.putText(pic, item.name, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1) # White background
                    if item.dept != 0: # Draws dept esitmation if there is one
                        cv2.putText(pic, f'Distance:{int(item.dept)} cm',item.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
                        cv2.putText(pic, f'Distance:{int(item.dept)} cm',item.position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        if self.hud:
            self.draw_hud(pic)
    
    def draw_hud(self, pic):
        for a in range(-20,21, 5):
            off = int(self.sensor['gyro'][2]*20+a*20 + self.middley)
            if 0 < off < self.height:
                if(a%2==0):
                    length = self.length
                else:
                    length = self.length2
                cv2.line(pic, (self.right, off), (self.right+length, off), self.color, 2) # 20 deg right
                cv2.putText(pic, f'{a}', (int(self.right+length+10), int(off+5)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2) # 20 deg right text
                cv2.line(pic, (self.left-length, off), (self.left, off), self.color, 2) # 20 deg left
                #cv2.putText(pic, f'{a}', (self.left-length-45, off+5), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2) # 20 deg left text
        dept = self.sensor['gyro'][0]
        dept = 2400
        
        cv2.rectangle(pic, self.squarestart, self.squarestop, self.color, 2)
        cv2.putText(pic, f'Depth', (int(self.squarestart[0]-10) ,int(self.squarestart[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        
        offset = int(dept+10 - math.floor(dept/10)*10)
        dep_text = int(int(dept/10)*10)
        for a in range(100 , 0 , -10):
            cv2.line(pic, (int(self.squarestart[0]+4), int(self.squarestart[1]+a*3+offset)), (int(self.squarestop[0]-4), int(self.squarestart[1]+a*3+offset)), self.color, 2)
            if a != 50 and a != 0:
                space = len(f'{(a-50+dep_text)}')
                cv2.putText(pic, f'{(a-50+dep_text)}', (int(self.squarestart[0]-space*15-30), int(self.squarestart[1]+a*3+offset+5)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.color, 2)
        space = len(f'{(dept)}')
        
        points = np.array(self.cursor.get_points(self.sensor['gyro'][1]))
        cv2.polylines(pic, [points], False, (0,0,255), 2)
        cv2.putText(pic, f'{dept}', (int(self.squarestart[0]-space*15-30), int(self.center[1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.line(pic, (int(self.squarestart[0]+4), int(self.center[1])), (int(self.squarestop[0]-4), int(self.center[1])), (0,0,255), 2)

    def update_data(self, sens):
        self.sensor = sens


def find_horizontal_line(matri): # Returns the ypos with the most white pixels from a bitmap picture
    big = 200000
    ypos = 0
    for a in range(int(len(matri/10))):
        temp = np.sum(matri[a*10:10+a*10])
        if temp > big:
            big = temp
            ypos = 10*a
    if not ypos==0:
        print(big)
        return ypos
    else:
        return False

def find_vertical_line(matri): # Returns the xpos with the most white pixels from a bitmap picture
    big = 200000
    xpos = 0
    for a in range(int(len(matri[0]/10))):
        temp = np.sum(matri[:, a*10:10+a*10])
        if temp > big:
            big = temp
            xpos = 10*a
    if not xpos == 0:
        return xpos
    else:
        return False

def to_bitmap(pic, masks):
        pic = cv2.GaussianBlur(pic,(25,25),0) # Filter image for better results
        hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, masks[0], masks[1])
        mask2 = cv2.inRange(hsv, masks[2], masks[3])
        _, tempblack = cv2.threshold(pic, 70, 255, cv2.THRESH_BINARY)
        canvas = np.zeros(tempblack.shape, np.uint8) # Heil svart maske
        mas2 = cv2.bitwise_or(mask, mask2, canvas)
        kernel = np.ones((5,5),np.uint8)
        mas2 = cv2.dilate(mas2, kernel, iterations = 1)
        return mas2

def merd_yaw(pic1, pic2, orb):
        pic1 = cv2.GaussianBlur(pic1,(25,25),0)
        pic2 = cv2.GaussianBlur(pic2,(25,25),0)
        pic1 = cv2.cvtColor(pic1, cv2.COLOR_BGR2GRAY)
        pic2 = cv2.cvtColor(pic2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(pic1 ,None)
        kp2, des2 = orb.detectAndCompute(pic2 ,None)
        failed = False
        a92 = 0
        try:
            mached_pixels = bf.match(des1, des2)
        except Exception as e:
            failed = True
        if not failed:
            mached_pixels = sorted(mached_pixels, key = lambda x:x.distance)
            #print(len(mached_pixels))
            dobbel_list = [[],[]]
            if len(mached_pixels) > 2:
                for a in mached_pixels:
                    if abs(kp1[a.queryIdx].pt[1] - kp2[a.trainIdx].pt[1]) < 15:
                        pic1 = cv2.circle(pic1, (int(kp1[a.queryIdx].pt[0]), int(kp1[a.queryIdx].pt[1])), 4, (255,0,0), -1)
                        pic1 = cv2.circle(pic1, (int(kp2[a.trainIdx].pt[0]), int(kp2[a.trainIdx].pt[1])) , 4, (255,0,0), -1)
                        if kp1[a.queryIdx].pt[0] < 640:
                            dobbel_list[0].append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]))
                        else:
                            dobbel_list[1].append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]))
            if len(dobbel_list[0]) > 1 and len(dobbel_list[1]) > 1:
                a92 = statistics.mean(dobbel_list[0])-statistics.mean(dobbel_list[1])
            return a92
        else:
            return False

class Maske_gui(QtWidgets.QDialog, Ui_Auto_focus_settings):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.parameters = {
            'h1':0 ,
            's1':0 ,
            'v1':0 ,
            '1_upper':0 ,
            'h2':0 ,
            's2':0 ,
            'v2':0 ,
            '2_upper':0 ,
        }
        self.Pink_slider_1.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_1, self.Pink_spin_1, 'h1'))
        self.Pink_slider_2.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_2, self.Pink_spin_2, 's1'))
        self.Pink_slider_3.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_3, self.Pink_spin_3, 'v1'))
        self.Pink_slider_4.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_4, self.Pink_spin_4, '1_upper'))
        self.Pink_slider_5.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_5, self.Pink_spin_5, 'h2'))
        self.Pink_slider_6.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_6, self.Pink_spin_6, 's2'))
        self.Pink_slider_7.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_7, self.Pink_spin_7, 'v2'))
        self.Pink_slider_8.actionTriggered.connect(lambda: self.change_value(self.Pink_slider_8, self.Pink_spin_8, '2_upper'))
        self.Pink_spin_1.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_1, self.Pink_slider_1, 'h1'))
        self.Pink_spin_2.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_2, self.Pink_slider_2, 's1'))
        self.Pink_spin_3.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_3, self.Pink_slider_3, 'v1'))
        self.Pink_spin_4.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_4, self.Pink_slider_4, '1_upper'))
        self.Pink_spin_5.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_5, self.Pink_slider_5, 'h2'))
        self.Pink_spin_6.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_6, self.Pink_slider_6, 's2'))
        self.Pink_spin_7.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_7, self.Pink_slider_7, 'v2'))
        self.Pink_spin_8.valueChanged.connect(lambda: self.change_value_spin(self.Pink_spin_8, self.Pink_slider_8, '2_upper'))
        self.img = cv2.imread('blank.jpg')
    
    def change_value(self, slider, spin, index, div=0):
        div = 10 ** div
        if div == 1:
            self.parameters[index] = slider.sliderPosition()
        else:
            self.parameters[index] = slider.sliderPosition() / div
        spin.setValue(self.parameters[index])

    def change_value_spin(self, spin, slider, index, mult=0):
        mult = 10 ** mult
        self.parameters[index] = spin.value()
        if mult == 1:
            slider.setSliderPosition(int(self.parameters[index]))
        else:
            slider.setSliderPosition(int(self.parameters[index] * mult))
        self.mask_pic()
    
    def mask_pic(self):
        lower_2 = np.array([self.parameters['h1'],self.parameters['s1'],self.parameters['v1']])
        upper_2 = np.array([self.parameters['1_upper'],255,255])
        lower_red = np.array([self.parameters['h2'],155,121])
        upper_red = np.array([180,255,255])
        masks = [lower_2, upper_2, lower_red, upper_red]
        img = to_bitmap(self.img, masks)
        cv2.imshow('frame', img)
        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows() # destroys the window showing image

def main1():
    c = Camera(1)
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True )
    a92, sta1, sta2 = 0,0,0
    lower_2 = np.array([0,80,140])
    upper_2 = np.array([18,255,255])
    lower_red = np.array([166,155,121])
    upper_red = np.array([180,255,255])
    masks = [lower_2, upper_2, lower_red, upper_red]
    while True:
        failed = False
        b1, b2 = c.aq_image(True)
        b1 = cv2.GaussianBlur(b1,(25,25),0)
        hsv = cv2.cvtColor(b1, cv2.COLOR_BGR2HSV)
        b1 = cv2.GaussianBlur(b1,(25,25),0)
        b2 = cv2.GaussianBlur(b2,(25,25),0)
        b12 = cv2.cvtColor(b1, cv2.COLOR_BGR2GRAY)
        b22 = cv2.cvtColor(b2, cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(b12 ,None)
        kp2, des2 = orb.detectAndCompute(b22 ,None)
        try:
            mached_pixels = bf.match(des1, des2)
        except Exception as e:
            failed = True
        if not failed:
            mached_pixels = sorted(mached_pixels, key = lambda x:x.distance)
            #print(len(mached_pixels))
            dobbel_list = [[],[]]
            if len(mached_pixels) > 2:
                for a in mached_pixels:
                    if abs(kp1[a.queryIdx].pt[1] - kp2[a.trainIdx].pt[1]) < 15:
                        b12 = cv2.circle(b12, (int(kp1[a.queryIdx].pt[0]), int(kp1[a.queryIdx].pt[1])), 4, (255,0,0), -1)
                        b22 = cv2.circle(b22, (int(kp2[a.trainIdx].pt[0]), int(kp2[a.trainIdx].pt[1])) , 4, (255,0,0), -1)
                        if kp1[a.queryIdx].pt[0] < 640:
                            dobbel_list[0].append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]))
                        else:
                            dobbel_list[1].append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]))
            if len(dobbel_list[0]) > 1 and len(dobbel_list[1]) > 1:
                sta1 = sta1*0.9 + 0.1*statistics.mean(dobbel_list[0])
                sta2 = sta2*0.9 + 0.1*statistics.mean(dobbel_list[1])
                a92 = sta1-sta2
            a92= int(a92*0.95+0.05*(sta1-sta2))
            #print(f"{a92}\t{int(sta1)}\t{int(sta2)}")
            #print(type(statistics.mean(dobbel_list[0])))
            #print(statistics.mean(dobbel_list[0]))
            #print(statistics.mean(dobbel_list[1]))
        #mask = cv2.inRange(hsv, lower_red, upper_red)
        #mask2 = cv2.inRange(hsv, lower_2, upper_2)
        #hsv = cv2.Canny(hsv, 50, 100, apertureSize=3)
        #linjer = cv2.HoughLinesP(hsv, 1, np.pi/180, 100)
        #print(type(linjer))
        #if isinstance(linjer, np.ndarray):
        #    print(len(linjer))
        #if linjer is not None:
        #    for i in range(0, len(linjer)):
        #        l = linjer[i][0]
        #        cv2.line(b1, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)
        #print(len(linjer))
        #print(len(dif_list))
        #for a in dif_list:
        #    print(f'{a}\n')
        #b12 = cv2.threshold(b12, 127, 255, cv2.THRESH_BINARY)
        #asd = cv2.findContours(b12, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(b1, asd, -1, (255,0,0), 2)
        #_, tempblack = cv2.threshold(b12, 70, 255, cv2.THRESH_BINARY)
        #canvas = np.zeros(tempblack.shape, np.uint8) # Heil svart maske

        #st = cv2.StereoBM_create(numDisparities=16, blockSize=5)
        #st.setTextureThreshold(256)
        #disparity = st.compute(b12,b22)

        #mas2 = cv2.bitwise_or(mask, mask2, canvas)
        #kernel = np.ones((5,5),np.uint8)
        #mas2 = cv2.dilate(mas2, kernel, iterations = 1)
        #print(find_vertical_line(mas2))
        #print(yaw(b1, b2, orb))
        mas2 = to_bitmap(b1, masks)
        hline = find_horizontal_line(mas2)
        vline = find_vertical_line(mas2)
        if hline:
            cv2.line(b1, (0,hline), (1280, hline), (255,255,0), 3)
        if vline:
            cv2.line(b1, (vline, 0), (vline,720), (255,255,0), 3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #plt.imshow(disparity)
        #cv2.imshow('asd', mas2)
        cv2.imshow('asasdd', b1)

        #plt.show()
    cv2.destroyAllWindows()

def main2():
    app = QApplication(sys.argv)
    win = Maske_gui()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main2()