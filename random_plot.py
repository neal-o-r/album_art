import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob


def pick_images():

        dirs = glob.glob('data/train/*')
        imgs = []
        for d in dirs:
                fs = glob.glob(f'{d}/*')
                f = np.random.choice(fs)
                imgs.append(f)

        return imgs

def plot_images():

        imgs = pick_images()

        fig = plt.figure()
        columns = 5
        rows = 2
        for i in range(1, columns*rows +1):
                path = imgs[i -1]
                genre = path.split('/')[2]

                img = Image.open(path)

                fig.add_subplot(rows, columns, i)
                plt.imshow(img)
                plt.title(genre)
                plt.axis('off')
                plt.show()

