from inspect import currentframe, getframeinfo
import subprocess
import math
import cv2
import statistics
import numpy as np

# Print info om nåværende linje.
def ln(melding:str=""):
    cf = currentframe()
    print(f"Linjenummer: {cf.f_back.f_lineno}, {melding + ', ' if melding != '' else ''}fil: { getframeinfo(cf.f_back).filename.split('/')[-1] }")

def get_apu_temp(string_out=True):
    temperature = subprocess.check_output("sensors | grep \"Tdie\" | tr -s ' ' " " | cut -d ' ' -f 2", shell=True)[:-1].decode('utf-8')
    if string_out:
        return temperature
    else:
        return float(temperature.replace('°C', ''))


def calc_distance(dist, focal_len=400, camera_space=60): # Calculates distance to object using test data, needs position on object in two pictures
    """Regner ut distansen til et objekt. for stereo kamera

    Args:
        centers (_type_): Senterkoortdinat til objektet i begge bildene
        focal_len (float, optional): Focallength oppgitt i pixler. Defaults to 33.2.
        camera_space (int, optional): Distansen mellom kameraene i mm. Defaults to 60.

    Returns:
        int: Avstand i mm
    """
    #dist = abs(centers[0][0]-centers[1][0])
    if dist == 0:
        return 300
    #y =-0,000326267081189824000000000000x3 + 0,248885323144369000000000000000x2 - 61,946537053035200000000000000000x + 5 155,964477808620000000000000000000
    return int((-0.000326267081189824)*(dist**3) + 0.248885323144369*(dist**2) - 61.9465370530352*dist + 5155.96447780862) # cm
    #return int((3.631e-6 * (dist**4)) - (0.003035 * (dist**3)) + (0.9672 * (dist**2)) - (139.9 * dist) + 7862)
    #return int(((focal_len*camera_space)/dist))

class Cursor: #______/\______
    def __init__(self, line_len:int, spacing:int, offset:tuple) -> None: 
        self.line_len = line_len
        self.spacing = spacing
        self.offset = offset # (xpos, ypos) Normally center in a picture
        self.points = [
            (self.offset[0]-self.spacing-self.line_len, self.offset[1]),
            (self.offset[0]-self.spacing, self.offset[1]),
            (self.offset[0], self.offset[1]-self.spacing),
            (self.offset[0]+self.spacing, self.offset[1]),
            (self.offset[0]+self.spacing+self.line_len, self.offset[1])
        ]


    def rotate_point(self, x, y, angelrad):

        x -= self.offset[0]
        y -= self.offset[1]
        x1 = x
        y1 = y

        x = x1*math.cos(angelrad) - y1*math.sin(angelrad)
        y = x1*math.sin(angelrad) + y1*math.cos(angelrad)

        x += self.offset[0]
        y += self.offset[1]
        return [int(x),int(y)]

    def get_points(self, angel):
        point_list = []
        angelrad = math.radians(angel)
        for a in self.points:
            point_list.append(self.rotate_point(a[0], a[1], angelrad))
        return point_list


def find_horizontal_line(matri): # Returns the ypos with the most white pixels from a bitmap picture
    big = 200000
    ypos = 0
    for a in range(int(len(matri/10))):
        temp = np.sum(matri[a*10:10+a*10])
        if temp > big:
            big = temp
            ypos = 10*a
    if not ypos==0:
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
                        #pic1 = cv2.circle(pic1, (int(kp1[a.queryIdx].pt[0]), int(kp1[a.queryIdx].pt[1])), 4, (255,0,0), -1) # Testdraw cirlces
                        #pic1 = cv2.circle(pic1, (int(kp2[a.trainIdx].pt[0]), int(kp2[a.trainIdx].pt[1])) , 4, (255,0,0), -1) # Testdraw cirlces
                        if kp1[a.queryIdx].pt[0] < 640:
                            dobbel_list[0].append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]))
                        else:
                            dobbel_list[1].append(abs(kp1[a.queryIdx].pt[0] - kp2[a.trainIdx].pt[0]))
            if len(dobbel_list[0]) > 1 and len(dobbel_list[1]) > 1:
                a92 = statistics.mean(dobbel_list[0])-statistics.mean(dobbel_list[1])
            return a92
        else:
            return False

class AutoMerd:
    def __init__(self, speed:int, pidK:float, resol:tuple,startdir:bool=True) -> None: 
        self.speed = speed
        self.right = startdir
        self.horisontal = True
        self.deadzone = 10
        self.ycenter = resol[1]/2
        self.xcenter = resol[0]/2
        self.offset_mult = pidK

    def new_data(self, xpos, ypos):
        vect = (0,0)
        if self.horisontal: # Horisontal direction, compensation for above/under rope
            if ypos:
                temp = ypos - self.ycenter
                if temp <= self.deadzone:
                    temp = 0
                if xpos:
                    if self.right:
                        if xpos > self.xcenter:
                            self.horisontal = False
                            #self.new_data(xpos, ypos)
                        else:
                            vect = (self.speed, temp) # Velocity vector
                    else:
                        if xpos < self.xcenter:
                            self.horisontal = True
                            #self.new_data(xpos, ypos)
                        else:
                            vect = (-self.speed, temp * self.offset_mult) # Velocity vector
                else:
                    vect = (self.speed if self.right else -self.speed, temp * self.offset_mult) # Velocity vector
                return vect
            else:
                return False
        else: # Vertical direction, compensation for disparity left/right of rope
            if xpos:
                temp = xpos - self.xcenter
                if xpos < self.deadzone:
                    temp = 0
                if ypos:
                    if ypos > self.ycenter:
                        self.horisontal = True
                        self.new_data(xpos, ypos)
                vect = (temp * self.offset_mult, self.speed)
            else:
                return False
            

