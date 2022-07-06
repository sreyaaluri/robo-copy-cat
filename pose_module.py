# code from: https://www.analyticsvidhya.com/blog/2021/05/pose-estimation-using-opencv/

import cv2
import mediapipe as mp
import time

class PoseDetector:

  def __init__(self, mode = False, model_complexity = 1, smooth_lm = True, segment = False, smooth_seg = True, detection_con = 0.5, track_con = 0.5):
    self.mode = mode
    self.model_complexity = 1
    self.smooth_lm = smooth_lm
    self.segment = segment
    self.smooth_seg = smooth_seg
    self.detection_con = detection_con
    self.track_con = track_con
    self.mp_draw = mp.solutions.drawing_utils # mediapipe's draw
    self.mp_pose = mp.solutions.pose          # mediapipe's pose
    self.pose = self.mp_pose.Pose(self.mode, self.model_complexity, self.smooth_lm, self.segment, self.smooth_seg, self.detection_con, self.track_con)

  def find_pose(self, img, draw=True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)
    #print(results.pose_landmarks)
    if self.results.pose_landmarks:
      if draw:
        self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
    return img

  def get_position(self, img, draw=True):
    lmList = []
    if self.results.pose_landmarks:
      for id, lm in enumerate(self.results.pose_landmarks.landmark):
        h, w, c = img.shape
        #print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        # lmList.append([id, cx, cy])
        lmList.append([id, lm.x, lm.y, lm.z])
        if draw and id == 5:
          cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return lmList

def main():
  cap = cv2.VideoCapture(0) # make VideoCapture(0) for webcam else 'videos/a.mp4'
  pTime = 0
  detector = PoseDetector()
  while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lmList = detector.get_position(img)
    for lm in lmList:
      print(lm)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main() 