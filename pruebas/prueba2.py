import cv2

img =cv2.imread('imagenes/18871.jpeg',0)

surf =cv2.xfeatures2d.SURF_create(400)

kp,des =surf.detectAndCompute(img,None)

print(len(kp))