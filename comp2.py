import cv2
import numpy as np

original = cv2.imread("imagenes/18871.jpeg")
duplicada = cv2.imread("imagenes/18871.jpeg")

# Checamos si son iguales

if original.shape == duplicada.shape:
    print(" Las imagenes tienen el mismo tamaño y cantidad de canales")
    diferencia = cv2.subtract(original, duplicada)
    cv2.imshow("Diferencia", diferencia)

    b, g, r = cv2.split(diferencia)

#    cv2.imshow("r", r)
#    cv2.imshow("g", g)
#    cv2.imshow("b", b)

    print(cv2.countNonZero(b))

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("Las imagenes son completamente iguales")

cv2.imshow("Original", original)
cv2.imshow("Duplicada", duplicada)
cv2.waitKey(0)
cv2.destroyAllWindows()
