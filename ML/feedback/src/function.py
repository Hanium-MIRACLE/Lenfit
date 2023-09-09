import numpy as np

# 코사인 유사도 공식으로 각도 구하기
def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    
    return angle

# 각도의 한계에 따라서 피드백을 주는 함수
def feedback(angle, limit, comment):
    if angle > limit:
        return "Bad" # comment 추가
    else:
        return 'Good'
    

if __name__ == '__main__':
    fb = feedback(90, 90, 'example')
    print(fb)