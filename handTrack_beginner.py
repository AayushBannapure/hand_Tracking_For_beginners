'''
cv2 - Video Opperessions
mediapipe - Dev. by Google and is pretty powerful ML module
'''
import cv2
import mediapipe as mp
import time, keyboard


def main():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) #Static_image_mode to False as we are using a video and other are self explanatory
    mpDraw = mp.solutions.drawing_utils #We have to declare those for us to be able to draw on the video.
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0) # This is to init Video Recording
    while True: # Loop required for a loop 
        img = cap.read() #reading the captured video in realtime 
        img = cv2.flip(img, 1) # Fliping it on the x axis as to avoid mirror effect
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Changing BGR to RGB
        results = hands.process(imgRGB) # leaving the recognition on the AI
    #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    h, w, c = img.shape #Deciding the shape of the landmarks
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    #if id ==0:
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED) #Adding circles to indicate points
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) #Drawing the landmarks
        cTime = time.time() # time expressed in seconds since the epoch to a string representing local time
        fps = 1/(cTime-pTime) # for measuring serial and parallel execution, and comparing the results.
        pTime = cTime
        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,255), 3) #Deciding the fps 
        cv2.imshow("Image", img) # Shows the video with annotations on the screen
        cv2.waitKey(1)
        if keyboard.is_pressed("q"): # Detecting if "q" is pressed if true exiting the code else running without disturbance
            break

if __name__ == "__main__":
    main()