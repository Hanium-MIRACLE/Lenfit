from src.object import *
from src.func import *


'''
Result 저장 형식

이름 /
    날짜 1/
        운동 이름 1/
            ...
        운동 이름 2/
            ...
        ...
    날짜 2/
        운동 이름 1/   
            ...
        운동 이름 2/   
            ...
        ...
    ...

'''



# main
if __name__ == '__main__':

    name = 'minsoo'
    input_video_dir = f'data/test_img/{name}'       # input video path

    minsoo = Person(name = name, 
                    preferred_fitness = ['squat', 'pushups'], 
                    input_video_dir = input_video_dir)


    minsoo.analyze_fitness(fitness='pushups', mode='Video')
    