import multiprocessing
from multiprocessing.dummy import Array
from mercury import Mercury
from mercury import venus
from theia import Theia
import time
import threading
import json

# Our main loop for both programs
def main_loop():
    ip = "127.0.0.1"
    m = Mercury()
    #m.toggle_network()
    #m.toggle_USB
    m.thei.toggle_front()
    m.thei.toggle_back()

    while(1):
        #if not t.camera_status[0]:
        #    print("Camera top stopped")
        #elif not t.camera_status[1]:
        #    print("Camera back stopped")
        time.sleep(2)
        print("Still running")

if __name__ == "__main__":
    main_loop()