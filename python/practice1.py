# 과제1: 구구단 + 함수 + 배열 + 사전
# 콘솔에 "몇 단을 출력하고 싶은지 숫자를 쓰세요. 100을 입력하면 종료됩니다: " 라고 출력된다
# 사용자가 3을 입력한 경우 3단이 출력된다
# 사용자가 0을 입력한 경우 구구단 전체가 출력된다
# 엉뚱한 값을 입력하면 잘못 입력했다고 예외를 출력한다
# 출력한 후에 다시 "몇 단을 출력하고 싶은지 숫자를 쓰세요. 100을 입력하면 종료됩니다: "로 되돌아 간다
# 100을 입력하면 프로그램이 종료된다

def make_googoodan(n):
    for i in range(1, 10):
        print("{} x {} = {}".format(n, i, n*i))

while True:
    num = int(input('몇 단을 출력하고 싶은지 숫자를 쓰세요. 100을 입력하면 종료됩니다: '))

    if num == 0:
        for i in range(2,10):
            make_googoodan(i)

    elif num == 100:
        break

    else:
        make_googoodan(num)

quit()