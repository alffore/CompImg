import cv2
import numpy as np

original_color = cv2.imread("imagenes/duplicada.jpg")
imagen_comprar_color = cv2.imread("imagenes/18871.jpeg")

original = cv2.cvtColor(original_color, cv2.COLOR_BGR2GRAY)
imagen_comprar = cv2.cvtColor(imagen_comprar_color, cv2.COLOR_BGR2GRAY)

# Checamos si son iguales

if original.shape == imagen_comprar.shape:
    print(" Las imagenes tienen el mismo tamaño y cantidad de canales")
    diferencia = cv2.subtract(original, imagen_comprar)

    b, g, r = cv2.split(diferencia)

    if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
        print("Las imagenes son completamente iguales")
    else:
        print("Las imagenes NO son completamente iguales")

# Checamos similaridad

orb = cv2.ORB_create()

kp_1 = orb.detect(original, None)
kp_2 = orb.detect(imagen_comprar, None)

kp_1, desc_1 = orb.compute(original, kp_1)
kp_2, desc_2 = orb.compute(imagen_comprar, kp_2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc_1, desc_2)

good = []

for i, m in enumerate(matches):
    if i < len(matches) - 1 and m.distance < 0.7 * matches[i + 1].distance:
        good.append(m)


print("Puntos 1: "+str(len(kp_1)))
print("Puntos 2: "+str(len(kp_2)))

print("Matches: "+str(len(good)))


resultado = cv2.drawMatches(original, kp_1, imagen_comprar, kp_2, good, None)
cv2.imshow("Resultado", resultado)





cv2.imshow("Original", original)
cv2.imshow("Duplicada", imagen_comprar)
cv2.waitKey(0)
cv2.destroyAllWindows()
