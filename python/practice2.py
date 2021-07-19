import random
import csv
import pandas as pd

while True:
    try:
        select = int(input("입력모드(1), 시험모드(2), 프로그램 종료(0) 중 하나를 선택하세요: "))
        if select == 1:
            f = open("voca.csv", "w", newline="", encoding='utf-8')
            fw = csv.writer(f)
            while True:
                word = []
                eng = input("Write English word: ")
                if eng == "exit":
                    f.close()
                    break
                kor = input("Write Korean word: ")
                word.append(eng)
                word.append(kor)
                fw.writerow(word)
        elif select == 2:
            eng_word = []
            kor_word = []
            f = pd.read_csv("voca.csv", header=None, encoding='utf-8')
            for i in range(len(f)):
                eng_word.append(f[0][i])
                kor_word.append(f[1][i])
            while True:
                rand_num = random.randrange(0, len(eng_word))
                test = input(f"{eng_word[rand_num]}의 뜻을 쓰세요: ")
                if test == kor_word[rand_num]:
                    print("맞았습니다!")
                elif test == "시험 끝":
                    print("시험이 끝났습니다.")
                    break
                else:
                    print("틀렸습니다!")
        elif select == 0:
            quit()
    except ValueError:
        print("Error")
