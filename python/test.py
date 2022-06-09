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
import threading
from yolo_detect import Yolo

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

class Athena(): 
    def __init__(self) -> None:
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True )
        self.first = True
        self.old_object_list = []
        self.first_width = True
        self.old_width_list = []
        
    # Diffrent methods to compare pixels in multiple pictures
    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)
    # 1
    #sift = cv2.SIFT_create()
    
    # 2
    #orb = cv2.ORB_create()
    #bf = cv2.BFMatcher()# OLD VERSION, THX OPENCV
    #bf = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=True )

    # 3
    #cv2.FlannBasedMatcher(index_paralgorithm = 1, trees = 5, checks = 50) # index_paralgorithm = FLANN_INDEX_KDTREE = 1
    def check_last_size(self, new_object_list):
        if self.first:
            self.first = False
            self.old_object_list = new_object_list
            return new_object_list
        if len(new_object_list) == len(self.old_object_list):
            for a, obj in enumerate(new_object_list): # Checks each object if its within 20% of old size and position
                if self.old_object_list[a].width*0.8 < obj.width < self.old_object_list[a].width*1.2:
                    #if self.old_object_list[a].position[0]*0.8 < obj.position[0] < self.old_object_list[a].position[0]*1.2:
                    if obj.dept <= 0:
#                        ln(f"{obj.dept}, {self.old_object_list[a].dept}")
                        obj.dept = self.old_object_list[a].dept
                    elif self.old_object_list[a].dept <= 50:
                        pass
                    else:
 #                       ln(f"{obj.dept}, {self.old_object_list[a].dept}")
                        obj.dept = self.old_object_list[a].dept*0.8 + obj.dept*0.2
        elif len(new_object_list) == 0 and len(self.old_object_list) != 0:
            return self.old_object_list
        self.old_object_list = new_object_list
        return self.old_object_list

    def check_width(self, new_object_list:list):
        if self.first_width:
            self.old_width_list = new_object_list
            return new_object_list
        else:
            if len(new_object_list) == len(self.old_width_list):
                for a, obj in enumerate(new_object_list):
                    if self.old_object_list[a].name == obj.name:
                        if self.old_object_list[a].width*0.8 < obj.width < self.old_object_list[a].width*1.2:
                            if obj.true_width <= 0:
                                obj.true_width = self.old_object_list[a].true_width
                            else:
                                obj.true_width = self.old_object_list[a].true_width*0.8 + obj.true_width*0.2

    def compare_pixles(self, object_list1, object_list2, pic):
        gray = [cv2.cvtColor(pic[0], cv2.COLOR_BGR2GRAY), cv2.cvtColor(pic[1], cv2.COLOR_BGR2GRAY)]
        new_object_list = []
        for obj1 in object_list1:
            for obj2 in object_list2:
                if obj1.position[1]-100 <= obj2.position[1] <= obj1.position[1]+100:
                    if obj1.width-50 <= obj2.width <= obj1.width+50:
                        points = [] # points to crop
                        points.append(int(obj1.rectangle[0][1]-obj1.height*0.1)) 
                        points.append(int(obj2.rectangle[0][1]-obj2.height*0.1)) 
                        points.append(int(obj1.rectangle[0][1]+obj1.height*1.1)) 
                        points.append(int(obj2.rectangle[0][1]+obj2.height*1.1)) 
                        points.append(int(obj1.rectangle[0][0]-obj1.width*0.1)) 
                        points.append(int(obj2.rectangle[0][0]-obj1.width*0.1)) 
                        points.append(int(obj1.rectangle[0][0]+obj1.width*1.1)) 
                        points.append(int(obj2.rectangle[0][0]+obj1.width*1.1)) 
                        for b, a in enumerate(points): # Checks that all points are within picture before crop
                            if a < 0:
                                points[b] = 0
                            if b <= 3:
                                if a > 720:
                                    points[b] = 720
                            else:
                                if a > 1280:
                                    points[b] = 1280
                        crop1 = gray[0][points[0]:points[2], points[4]:points[6]]
                        crop2 = gray[1][points[1]:points[3], points[5]:points[7]]
                        offset = obj1.rectangle[0][0]- obj2.rectangle[0][0]

                        # Testprints
                        #print(f'pos0:{int(obj1.rectangle[0][0])}')
                        #print(f'pos1:{int(obj1.rectangle[0][1])}')
                        #a = int(obj1.rectangle[0][0]+obj1.height*0.2)-int(obj2.rectangle[0][0]+obj1.height*0.2)
                        #b = obj1.rectangle[0][0]- obj2.rectangle[0][0]
                        #print(f'{(a==b)}')
                        #print(f'Offset:{offset}')
                        #print(f'Width1:{obj1.width}, height1:{obj1.height}')
                        #print(f'Width2:{obj2.width}, height2:{obj2.height}')
                        #cv2.imshow("TAGE1!!!!", crop1)
                        #cv2.imshow("TAGE2!!!!", crop2)
                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #        break
                        
                        kp1, des1 = self.orb.detectAndCompute(crop1 ,None)
                        kp2, des2 = self.orb.detectAndCompute(crop2 ,None)
                        try:
                            mached_pixels = self.bf.match(des1, des2)
                            mached_pixels = sorted(mached_pixels, key = lambda x:x.distance)
                            print(len(mached_pixels))
                            new_list = []
                            for a in mached_pixels:
                                if a.distance < 100:
                                    new_list.append(a)
                        except Exception as i:
                            print(i)
                            return
                        mached_pixels = new_list
                        #imgDummy = np.zeros((1,1))
                        #img = cv2.drawMatches(crop1,kp1,crop2,kp2,mached_pixels[:10], imgDummy, flags=2)
                        #cv2.imshow("TAGE1!!!!", img)
                        dif_list = []
                        if len(mached_pixels) > 2:
                            for a in mached_pixels:
                                if abs(kp1[a.queryIdx].pt[1] - kp2[a.trainIdx].pt[1]) < 10:
                                    crop1 = cv2.circle(crop1, (int(kp1[a.queryIdx].pt[0]), int(kp1[a.queryIdx].pt[1])), 4, (255,0,0), -1)
                                    crop2 = cv2.circle(crop2, (int(kp2[a.trainIdx].pt[0]), int(kp2[a.trainIdx].pt[1])) , 4, (255,0,0), -1)
                                    dif_list.append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]+offset))
                            if len(dif_list) > 2:   
                                med = statistics.median(dif_list)
                                if 158 < med < 218:
                                    obj1.dept = calc_distance(med)
                                    ln(f"{obj1.dept}")
                                
                                # Print for calibration
                                #print(f'Disparity: {statistics.median(dif_list)}')
                                
                                #cv2.imshow("TAGE1!!!!", crop1)
                                #cv2.imshow("TAGE2!!!!", crop2)
                                #if cv2.waitKey(1) & 0xFF == ord('q'):
                                #    break
                                #pass
                                #print(f'Mean:{statistics.mean(dif_list)}')
                                #print(f'Median:{statistics.median(dif_list)}')
                        #plt.imshow(img),plt.show()
                        #cv2.imshow("TAGE2!!!!", crop2)
                        #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #    break
            new_object_list.append(obj1)
        if len(new_object_list) > 1:
            new_object_list = check_overlap(new_object_list) # Checks for object overlapping
        new_object_list = self.check_last_size(new_object_list) # Filters distances vales, and sets old if new is not found
        return new_object_list

def check_objects_calc_size(objects):
    rubber_list = {"rubberfish":[], "head":[], "tail":[]}
    out = []
    
    for a in objects:
        if str(a.name).lower() in rubber_list.keys():
            rubber_list[str(a.name).lower()].append(a)

    for key in rubber_list.keys():
        if len(rubber_list[key]) > 1:
            rubber_list[key] = check_overlap(rubber_list[key])
    if len(rubber_list["rubberfish"]) > 0 and len(rubber_list["head"]) > 0 and len(rubber_list["tail"]) > 0:
        for fish in rubber_list["rubberfish"]:
            temp_list = []
            for head in rubber_list["head"]:
                if is_overlap(head.position, fish.rectangle):
                    temp_list.append(head)
                    break
            if len(temp_list) > 0:
                for tail in rubber_list["tail"]:
                    if is_overlap(tail.position, fish.rectangle):
                        temp_list.append(tail)
                        break
            if len(temp_list) == 2:
                fish.true_width = calc_size_fish(temp_list)

    return rubber_list["rubberfish"]

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
            's1':80 ,
            'v1':140 ,
            '1_upper':18 ,
            'h2':166 ,
            's2':155 ,
            'v2':121 ,
            '2_upper':180 ,
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
        self.startup_positons()
        threading.Thread(name="COM_cam_1",target=mask_pic, daemon=True, args=(self.parameters, self.img)).start()
    
    def startup_positons(self):
        self.Pink_spin_1.setValue(self.parameters['h1'])
        self.Pink_spin_2.setValue(self.parameters['s1'])
        self.Pink_spin_3.setValue(self.parameters['v1'])
        self.Pink_spin_4.setValue(self.parameters['1_upper'])
        self.Pink_spin_5.setValue(self.parameters['h2'])
        self.Pink_spin_6.setValue(self.parameters['s2'])
        self.Pink_spin_7.setValue(self.parameters['v2'])
        self.Pink_spin_8.setValue(self.parameters['2_upper'])

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
    
def mask_pic(parameters, img):
    while True:
        temp = np.copy(img)
        lower_2 = np.array([parameters['h1'], parameters['s1'], parameters['v1']])
        upper_2 = np.array([parameters['1_upper'],255,255])
        lower_red = np.array([parameters['h2'],155,121])
        upper_red = np.array([180,255,255])
        masks = [lower_2, upper_2, lower_red, upper_red]
        temp = to_bitmap(temp, masks)
        cv2.imshow('frame', temp)
        cv2.waitKey(50) # waits until a key is pressed
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

def main3():
    time_list = []
    mode = 1 # 1: Find rubberfish, 2: mosaikk 3:TBA 
    #TODO Her skal autonom kjøring legges inn
    old_list = []
    first = True
    width = 1280
    lower_2 = np.array([0,80,140])
    upper_2 = np.array([18,255,255])
    lower_red = np.array([166,155,121])
    upper_red = np.array([180,255,255])
    masks = [lower_2, upper_2, lower_red, upper_red]
    st_list = [] # List of images to stitch
    ath = Athena()
    new_pic = False
    stitch = False
    orb = cv2.ORB_create()
    cap = cv2.VideoCapture('C:\\Skole\\video\\asd.mp4')
    while (cap.isOpened()):
        ret, img = cap.read()
        if first:
            first = False
            s = img.shape
            #yal = Yolo( (1024, 720) )
        if ret == True:
            #res1 = yal.yolo_image(img)
            #mached_list = check_objects_calc_size(res1)
            #mached_list = ath.check_width(mached_list)
            cv2.imshow('Frame',img)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()




            


if __name__ == "__main__":
    main3()