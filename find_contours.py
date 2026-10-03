import cv2
import math
import sys
import numpy as np

from geometry import detect_game_objects, line_intersects_contour

assert(len(sys.argv) == 2)

img_num = int(sys.argv[1])

img  = cv2.imread(f"out/frame{img_num}.jpg")

full_frame = img[:525,:]
mask_frame = img[525:,:]

enemies, foods, us, _ = detect_game_objects(full_frame,False)

cx = round(1265/2)
cy = round(525/2)

enemies.sort(key=lambda e: e.moment[1])
enemy = enemies[-2]
foods.sort(key=lambda f: math.hypot((f.moment[0] - cx), (f.moment[1] - cy)))
food = foods[0]
p1 = (cx, cy)
p2 = ((food.moment[0] - cx) * 1.3 + food.moment[0], (food.moment[1] - cy) * 1.3 + food.moment[1])
p2 = [round(p) for p in p2]

mask_frame = cv2.line(mask_frame, p1, p2, (0,0,255), 1)
print(line_intersects_contour(mask_frame.shape, enemy.contour,  p1, p2))

mask_frame = cv2.drawContours(mask_frame, enemy.contour, -1, (0,255,0),1)

cv2.imwrite(f"out/frame{img_num}contours.jpg", mask_frame)

