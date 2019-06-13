#
# https://pypi.org/project/ImageHash/
#
from PIL import Image
import imagehash

hash = imagehash.average_hash(Image.open('../imagenes/18871.jpeg'))
print(hash)

hash2 = imagehash.dhash(Image.open('../imagenes/18871.jpeg'),64)
print(hash2)

hash3 = imagehash.phash(Image.open('../imagenes/18871.jpeg'),64)
print(hash3)

hash4 = imagehash.whash(Image.open('../imagenes/18871.jpeg'),32)
print(hash4)