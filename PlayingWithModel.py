from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss
#Last but not least, now we are playing, actually our model is playing ^w^
#I will explain one by one again.

mon = {"top":460, "left":830, "width":270, "height":75}
sct = mss()

width = 125
height = 50

# Loading the model we created.
model = model_from_json(open("model_new.json","r").read())
model.load_weights("agackesme.h5")

# right = 0, left = 1 here
labels = ["Right", "Left"]
# This delay is important. If ur delay is little, maybe neural network won't catch the game.
framerate_time = time.time()
delay = 0.08
key_down_pressed = False

#We are recording screen (mon) and our model is reacting to this record.
#Basically model sees the screen and decides for pressing right or left. Then presses.
while True:
    
    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255
    
    X =np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    r = model.predict(X)
    
    result = np.argmax(r)
    
    
    if result == 1: # right = 0
        
        keyboard.press("Right")
        time.sleep(delay)
        
    elif result == 0:    # left = 1
        
        keyboard.press("Left")
        time.sleep(delay)
        
       
    
    




























