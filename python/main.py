#!/usr/bin/python3.10
# -*- coding: UTF-8 -*-

import multiprocessing
from multiprocessing.dummy import Array
from mercury import Mercury
from theia import Theia
import time
import threading
import json
import os
os.chdir(os.path.dirname(os.path.abspath(__file__))) # Bytter working directory til den nåværende slik at programmet kan startes utenfra mappa

# Our main loop for both programs
def main_loop():
    m = Mercury()
    m.thei.toggle_front()
    #m.thei.toggle_back()

    while(1):
        time.sleep(0.2)
        #m.ping()
        check = m.update_hud_data()
        if not check:
            print("Did not send data")
        m.auto_control()
        #time.sleep(0.7)
        #print("Still running.  ", end='\r')
        #time.sleep(0.7)
        #print("Still running.. ", end='\r')
        #time.sleep(0.7)
        #print("Still running...", end='\r')

if __name__ == "__main__":
    main_loop()
