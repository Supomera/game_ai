import keyboard
import uuid
import time
from PIL import Image
from mss import mss

#In this code, I'm trying to build my own dataset due to the game.
#So at this algorithm, first of all I'm saving a part of my screen (mon) when I pressed to keys.
#I'm setting the location of mon and then playing the game. Keys that I pressed and saved images are being matched by algorithm.


#Here I'm setting the location of mon.
mon = {"top":460, "left":830, "width":270, "height":75}
sct = mss()

i = 0



#Secondly I wrote a function for saving and matching photos of the game with pressed keys.
def record_screen(record_id, key):
    global i
    
    i += 1
    print("{}: {}".format(key, i))
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))
    
is_exit = False

def exit():
    global is_exit
    is_exit = True
    
keyboard.add_hotkey("esc", exit)

record_id = uuid.uuid4()

#There might be some problems about RuntimeError, so I've added try/except commands and blocked those possible problems.
while True:
    
    if is_exit: break

    try:
        if keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
        elif keyboard.is_pressed("left"):
            record_screen(record_id, "left")
            time.sleep(0.1)
        
    except RuntimeError: continue
            























