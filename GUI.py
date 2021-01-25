from tkinter import messagebox
from Neural_network import NeuralNetworkModel

import cv2
import numpy as np
from tkinter import *
from PIL import ImageGrab


def clear():
    global c
    c.delete("all")


def activate_event(event):
    global lastX, lastY
    c.bind('<B1-Motion>', draw_lines)
    lastX, lastY = event.x, event.y


def draw_lines(event):
    global lastX, lastY
    x, y = event.x, event.y
    c.create_line((lastX, lastY, x, y), width=8, fill='black',
                  capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastX, lastY = x, y


def recognize_digit():
    global image_number, neural_network
    predictions = []
    percentage = []
    filename = f'image{image_number}.jpg'
    widget = c

    x = root.winfo_rootx() + widget.winfo_x()
    y = root.winfo_rooty() + widget.winfo_y()
    x1 = x + widget.winfo_width()
    y1 = y + widget.winfo_height()

    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    fl = True
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        roi = th[y - top:y + h + bottom, x - left:x + w + right]
        try:
            img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        except Exception:
            messagebox.showerror("Error!", "you went out of frame. Try to clear the frame and start over")
            fl = False
            break
        img = img.reshape(1, 28, 28, 1)
        pred = neural_network.model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + '  ' + str(int(max(pred) * 100)) + '%'
        print(data)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, font_scale, color, thickness)

    if fl:
        cv2.imshow('Any key on keyboard will close this', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


root = Tk()
root.resizable(0, 0)
root.title("Handwritten digit recognition")

neural_network = NeuralNetworkModel()

lastX, lastY = None, None
image_number = 0

c = Canvas(root, width=640, height=480, bg='white')
c.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
c.bind('<Button-1>', activate_event)

btn_save = Button(text="Recognize", command=recognize_digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
btn_clear = Button(text="clear", command=clear)
btn_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()
