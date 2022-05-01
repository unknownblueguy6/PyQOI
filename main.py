from pyqoi import pyqoi
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

image = Image.open('qoi_logo.png')
np_image = np.asarray(image)
qh = pyqoi.qoi_header(*image.size, channels = len(image.mode))

pyqoi.qoi_write('test.qoi', np_image, qh)

img, qh = pyqoi.qoi_read('qoi_logo.qoi')

plt.imshow(img)
plt.show()

