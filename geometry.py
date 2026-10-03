import numpy as np
import cv2
from typing import Tuple, List

from objects import GameObject

# ─── HSV Color Ranges ──────────────────────────────────────────────────────────
ENEMY_LOWER = np.array([0, 120, 120])
ENEMY_UPPER = np.array([179, 255, 255])
BG_LOWER = np.array([82, 50, 130])
BG_UPPER = np.array([99, 145, 245])

def object_is_in_center(contour) -> bool:
    center_point = (round(1265 / 2), round(565 / 2))
    center_targets = [center_point,(round(1265 / 2)+2, round(565 / 2)),(round(1265 / 2), round(565 / 2)+2),(round(1265 / 2)-2, round(565 / 2)),(round(1265 / 2), round(565 / 2)-2)]
    for point in center_targets:
        if cv2.pointPolygonTest(contour, point, False) >= 0:
            return True
    return False

def find_objects(
    mask: np.ndarray, min_area: int
) -> Tuple[List[GameObject], GameObject]:
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    us = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)

        if object_is_in_center(cnt):
            us = GameObject(
                moment=(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])),
                area=area,
                contour=cnt,
            )

        elif area > min_area:
            if M["m00"] > 0:
                object = GameObject(
                    moment=(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])),
                    area=area,
                    contour=cnt,
                )
                blobs.append(object)
    if us == None:
        print("did not find out own player, big problem")
    return blobs, us

def detect_game_objects(
    frame: np.ndarray, debug: bool
) -> Tuple[List[GameObject], List[GameObject], GameObject]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bg_mask = cv2.inRange(hsv, BG_LOWER, BG_UPPER)
    bg_mask = cv2.bitwise_not(bg_mask)
    # crop out score and scoreboard
    bg_mask[50:135, 550:725] = 0
    bg_mask[:275, 968:] = 0
    bg_mask = cv2.erode(bg_mask, np.ones((3, 3), np.uint8), iterations=5)

    min_area = 12
    objects, us = find_objects(bg_mask, min_area)
    area_cutoff = 1000

    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)
    if debug:
        for object in objects:
            bg_mask = cv2.circle(
                bg_mask,
                (object.moment[0], object.moment[1]),
                10,
                (0, 0, 255) if object.area > area_cutoff else (255, 0, 0),
            )
        # cv2.imwrite(f"{OUTPUT}/frame{tick}.jpg", np.vstack((frame, bg_mask)))

    objects.sort(key=lambda x: x.area)
    index_cutoff = len(objects)
    for i, object in enumerate(objects):
        if object.area > area_cutoff:
            index_cutoff = i - 1
            break

    food = objects[:index_cutoff]
    enemies = objects[index_cutoff:]

    return enemies, food, us, np.vstack((frame, bg_mask))



def segment_contour_intersections(p, q, contour):
    """Return an (M, 2) array of intersection points between segment p-q and a contour."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    pts = contour.reshape(-1, 2).astype(float)
    A, B = pts, np.roll(pts, -1, axis=0)   # edges A->B, closed loop

    r, s = q - p, B - A
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    denom = cross(r, s)                     # 0 means parallel
    ap = A - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = cross(ap, s) / denom            # position along the line segment
        u = cross(ap, r) / denom            # position along each contour edge

    hit = (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return p + t[hit, None] * r

def line_intersects_contour_bad(contour, p1, p2):
    return len(segment_contour_intersections(p1,p2,contour) > 0)


def line_intersects_contour(img_shape, contour, p1, p2):
    h, w = img_shape[:2]
    c_mask = np.zeros((h, w), np.uint8)
    l_mask = np.zeros((h, w), np.uint8)

    cv2.drawContours(c_mask, [contour], -1, 255, thickness=1)  # use -1 to treat the contour as a filled region
    cv2.line(l_mask, p1, p2, 255, thickness=4)  # thickness=2 avoids diagonal "slip-through" misses

    return cv2.countNonZero(cv2.bitwise_and(c_mask, l_mask)) > 0


