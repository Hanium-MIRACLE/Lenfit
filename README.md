# Lenfit
> pose estimation AI service


```$ conda create --name Lenfit python==3.8.15```

```$ conda activate Lenfit```

```$ pip install -r requirements.txt```



## Classification

* 사용자가 수행하고 있는 운동의 종류를 실시간으로 분류


## Count 

* 사용자의 운동 횟수를 실시간으로 count
* Pose estimation을 통해 Landmark 추출
* Bootstrap 을 이용하여 운동의 Up/Down을 분류하여 횟수를 Count
* 운동 Tempo를 추출


## feedback

* 관절의 각도를 측정하여 실시간으로 운동 자세에 대한 피드백 제공


## Score

* 전문가의 운동 영상으로 바탕으로 추출된 Tempo와 비교하여 사용자의 자세에 대한 점수를 측정