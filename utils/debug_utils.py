import matplotlib
import matplotlib.pyplot as plt


def show_image(image):
    plt.figure()
    plt.imshow(image)

    plt.axis('off')
    plt.ioff()
    # plt.pause(0.05)
    # plt.clf()
    plt.show()