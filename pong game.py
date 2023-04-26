import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4,720)

#importing all images
imgbackground = cv2.imread("C:\\Users\\USER\\PycharmProjects\\Projects\\Virtual Mouse\\Pong game\\Background.png")
imgGameover = cv2.imread("C:\\Users\\USER\\PycharmProjects\\Projects\\Virtual Mouse\\Pong game\\gameOver.png")
imgBall = cv2.imread("C:\\Users\\USER\\PycharmProjects\\Projects\\Virtual Mouse\\Pong game\\Ball.png",cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread("C:\\Users\\USER\\PycharmProjects\\Projects\\Virtual Mouse\\Pong game\\bat1.png",cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread("C:\\Users\\USER\\PycharmProjects\\Projects\\Virtual Mouse\\Pong game\\bat2.png",cv2.IMREAD_UNCHANGED)

print("background",imgbackground.shape)
print("gameover",imgGameover.shape)
print("ball",imgBall.shape)
print("bat1",imgBat1.shape)
print("bat2",imgBat2.shape)

scale_percent = 60

width = int(imgbackground.shape[1] * scale_percent / 100)
height = int(imgbackground.shape[1] * scale_percent / 100)

width = int(imgGameover.shape[1] * scale_percent / 100)
height = int(imgGameover.shape[1] * scale_percent / 100)

width = int(imgBall.shape[1] * scale_percent / 100)
height = int(imgBall.shape[1] * scale_percent / 100)

width = int(imgBat1.shape[1] * scale_percent / 100)
height = int(imgBat1.shape[1] * scale_percent / 100)

width = int(imgBat2.shape[1] * scale_percent / 100)
height = int(imgBat2.shape[1] * scale_percent / 100)

dim = (width, height)
resized1 = cv2.resize(imgbackground, dim, interpolation= cv2.INTER_AREA)
resized2 = cv2.resize(imgGameover, dim, interpolation= cv2.INTER_AREA)
resized3 = cv2.resize(imgBall, dim, interpolation= cv2.INTER_AREA)
resized4 = cv2.resize(imgBat1, dim, interpolation= cv2.INTER_AREA)
resized5 = cv2.resize(imgBat2, dim, interpolation= cv2.INTER_AREA)

print("background",resized1.shape)
print("gameover",resized2.shape)
print("ball",resized3.shape)
print("bat1",resized4.shape)
print("bat2",resized5.shape)


#Hand Detector
detector = HandDetector(detectionCon=0.8,maxHands=2)

#variables
ballPos = [100,100]
speedX = 15
speedY = 15
gameOver = False
score = [0,0]

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    #find the hand and it landmarks
    hands, img = detector.findHands(img, flipType = False)       #with draw

    #Overlaying the background images
    #img = cv2.addWeighted(img, 0.2,resized1,0.8, 0)

    #check for hands
    if hands:
        for hand in hands:
            x,y,w,h = hand['bbox']
            h1, w1, _ = resized4.shape
            y1 = y - h1 // 2
            y1 = np.clip(y1,20,415)

            if hand['type'] == "Left":
                img = cvzone.overlayPNG(img, resized4, (59, y1))
                if 59 < ballPos[0] < 59 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] += 30
                    score[0] += 1

            if hand['type'] == "Right":
                img = cvzone.overlayPNG(img, resized5, (1195, y1))
                if 1195-50 < ballPos[0] <1195 + w1 and y1 < ballPos[1] < y1 + h1:
                    speedX = -speedX
                    ballPos[0] -= 30
                    score[1] += 1

    #Game over
    if ballPos[0] < 40 or ballPos[0] > 1200:
        gameOver = True

    if gameOver:
        img = imgGameover
        cv2.putText(img, str(score[1] + score[0]).zfill(2),(585,360),cv2.FONT_HERSHY_COMPLEX,2.5,(200,0,200),5)

    #If game is not over move the ball
    else:
        #Move the ball
        if ballPos[1] >= 500 and ballPos[1] <= 10:
            speedY = -speedY

        ballPos[0] += speedX
        ballPos[1] += speedY

        #Draw the ball
        img = cvzone.overlayPNG(img, resized3, ballPos)

        cv2.putText(img, str(score[0]),(300,650),cv2.FONT_HERSHEY_SIMPLEX,3, (255,255,255), 5)
        cv2.putText(img, str(score[1]),(900,650),cv2.FONT_HERSHEY_SIMPLEX,3, (255,255,255), 5)

#    img[580:700, 20:233] = cv2.resize(imgRaw,(213,120))

    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        ballPos = [100,100]
        speedX = 15
        speedY = 15
        gameOver = False
        score = [0,0]
        imgGameover = cv2.imread("C:\\Users\\USER\\PycharmProjects\\Projects\\Virtual Mouse\\Pong game\\gameOver.png")
