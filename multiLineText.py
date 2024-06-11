import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from bidi.algorithm import get_display
import arabic_reshaper

def multi_line_text(wrapped_text, shift_up, font_size,thickness,color):
    img = np.zeros((600, 800, 3), dtype='uint8')
    height, width, channel = img.shape

    text_img = np.ones((height, width))
    font = cv2.FONT_HERSHEY_DUPLEX

    i = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]

        gap = textsize[1] + 20

        y = int(((img.shape[0] + textsize[1]) / 2) + shift_up) + i * gap
        x = int((img.shape[1] - textsize[0]) / 2)  # for center alignment => int((img.shape[1] - textsize[0]) / 2)

        cv2.putText(img, line, (x, y), font,
                    font_size,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA)
        i += 1
    return img

def arabic_text(text, fill, x,y,size):
    x=x
    y=y
    img = np.zeros((500, 700, 3), dtype='uint8')

    pil_image = Image.fromarray(img)
    font = ImageFont.truetype("C:\Windows\Fonts\\arial.ttf", size=size)
    draw = ImageDraw.Draw(pil_image)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    draw.text((x, y), bidi_text, font=font, fill=fill)
    image = np.asarray(pil_image)
    return image
