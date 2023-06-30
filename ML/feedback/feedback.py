# 각도의 한계에 따라서 피드백을 주는 함수

# input: angle, state, limit, comment
# output: comment

def angle_feedback(angle, state, limit, comment):
    if state == 'Squat':
        comment = feedback(angle, limit, comment)
    elif state == 'Pushup':
        pass
    return comment

# input: angle, limit, comment
# output: comment

def feedback(angle, limit, comment):
    if angle > limit:
        return comment
    else:
        return 'Good'