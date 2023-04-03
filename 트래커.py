import cv2
import csv
from matplotlib import pyplot as plt

isDragging = False
x0, y0, w, h = -1, -1, -1, -1
blue, red = (255, 0, 0), (0, 0, 255)
cnt = 0

def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, first_frame, roi, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y
    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            first_frame_draw = first_frame.copy()
            cv2.rectangle(first_frame_draw, (x0, y0), (x, y), blue, 2)
            cv2.imshow('first_frame', first_frame_draw)
    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0
            if w > 0 and h > 0:
                first_frame_draw = first_frame.copy()
                cv2.rectangle(first_frame_draw, (x0, y0), (x, y), red, 2)
                cv2.imshow('first_frame', first_frame_draw)
                roi = first_frame_gray[y0:y0+h, x0:x0+w]
                cv2.imshow('cropped', roi)
                cv2.imwrite('./cropped.png', roi)
            else:
                cv2.imshow('first_frame', first_frame)
                print('drag should start from left-top side')

video_file = './video/pendulum1-1.mp4'
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_file)
if cap.isOpened() :
    _, first_frame = cap.read()
else :
    print('fail')

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
frame = 30.0
delayms = int(1/frame*1000)
cntlist = []
objposx = []
objposy = []
velx = []
vely = []
accx = []
accy = []

cv2.imshow('first_frame', first_frame)
cv2.setMouseCallback('first_frame', onMouse)
cv2.waitKey()
cv2.destroyAllWindows()

if cap.isOpened():
    while cv2.waitKey(33) < 0 :
        ret, frame = cap.read()
        
        if ret:
            cnt+=1
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            res = cv2.matchTemplate(frame_gray, roi, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            mid = (top_left[0] + w/2, top_left[1] + h/2)
            cv2.rectangle(frame, top_left, bottom_right, red, 2)
            print(mid)
            objposx.append(mid[0])
            objposy.append(mid[1])
            cv2.imshow('detect', frame)
            cv2.waitKey(delayms)
            cntlist.append(cnt)
        else:
            break
        
else:
    print("can't open video.")
cap.release()
cv2.destroyAllWindows()

for i in range(cnt-1) :
    velx.append(objposx[i+1]-objposx[i])
    vely.append(objposy[i+1]-objposy[i])
for i in range(cnt-2) :
    accx.append(velx[i+1]-velx[i])
    accy.append(vely[i+1]-vely[i])


plt.subplot(3, 3, 1)
plt.scatter(objposx, objposy)
plt.subplot(3, 3, 2)
plt.scatter(cntlist, objposx)
plt.subplot(3, 3, 3)
plt.scatter(cntlist, objposy)

cntlist.pop()
plt.subplot(3, 3, 5)
plt.scatter(cntlist, velx)
plt.subplot(3, 3, 6)
plt.scatter(cntlist, vely)

cntlist.pop()
plt.subplot(3, 3, 8)
plt.scatter(cntlist, accx)
plt.subplot(3, 3, 9)
plt.scatter(cntlist, accy)
plt.show()

with open('object position.csv', 'w', newline='') as save :
    writer = csv.writer(save)
    for i in range(cnt) :
        writer.writerow([objposx[i], objposy[i]])