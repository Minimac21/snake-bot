import cv2
import sys
import numpy as np

from geometry import detect_game_objects

assert(len(sys.argv) == 2)

img_num = int(sys.argv[1])

img  = cv2.imread(f"out/frame{img_num}.jpg")

full_frame = img[:525,:]
mask_frame = img[525:,:]

enemies, foods, us, _ = detect_game_objects(full_frame,False)

for enemy in enemies:
	mask_frame = cv2.drawContours(mask_frame, enemy.contour, -1, (0,255,0),1)

cv2.imwrite(f"out/frame{img_num}contours.jpg", mask_frame)

