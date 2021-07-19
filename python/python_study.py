# 모든 키워드를 정확하게 알면 됩니다


# 1.
# 파이썬 파이참 컴파일언어 컴파일에러 인터프리터언어 런타임에러 pip 가상환경 디버깅

# python -m venv 가상환경
# 가상환경 내부의 activate.bat을 실행
# 파이참에서 activate.bat 실행을 유지하기 위해서 프로젝트 인터프리터 설정에서 가상환경 내부의 인터프리터를 선택

# pip install jupyter
# jupyter notebook

# pip freeze > requirements.txt
# pip install -r requirements.txt


# 2.
# 코멘트 자료형(문자 숫자 불린) 변수 콘솔 gui 프린트 포멧팅 형변환 연산자(= == != 사칙연산)


# 3.
# 사전 배열 튜플 집합
# 사전: 조회
# 배열: 조회, 변경, 추가, 삭제
# 튜플: 조회 가능, 변경, 추가, 삭제 불가
# 셋: 원소의 순서가 정해져 있지 않으며, 중복 불가능하고, 조회 불가, 집합 연산 가능


# 4.
# 함수 리턴

# 함수 사용시 주의사항(전역 = 현실 / 함수 = 꿈)
# 절차 지향적으로 프로그래밍 하는 경우 case 1 = 현실에서 현실을 바꾸면 당연히 바뀐다

# 절차 지향적으로 프로그래밍 하는 경우 case 2 = (꿈에서 결과를 얻은 후) 현실에서 현실을 바꾸면 당연히 바뀐다
# 전역 상태와 무관하게 받은 파라미터만 활용해서 연산하는 함수 = 순수함수
# => 순수함수로 값을 리턴 후 전역 변수에 재대입 => 파이썬 추천(메모리 문제 발생)

# 절차 지향적으로 프로그래밍 하는 경우 case 3 = 꿈에서 현실을 바꾸는 경우
# => 함수 내부에서 global 키워드를 쓴다 => 비추천

# 절차 지향적으로 프로그래밍 하는 경우 case 4 = 꿈에서 현실을 바꾸는 경우
# call by address(kind of call by value) => c++ 가능, 비추천
# call by reference => c++ 자바 가능, 추천(메모리 이익)

# 객체 지향적으로 프로그래밍을 하는 경우
# => 파이썬, c++ 모두 리턴 대신 인스턴스 변수를 변경한다


# 5.
# if elif else switch


# 6.
# while flag break for


# 7.
# 데이터가 복잡하면 관리하기 어렵다
# 복잡한 데이터를 잘 관리하는 방법 = 객체지향 프로그래밍

# 객체지향 프로그래밍의 4대 원칙
# 1. 추상화: 객체 사용자가 클래스 내부를 알지 못해도 편리하게 사용할 수 있도록 클래스를 잘 설계해야 한다
# 2. 캡슐화: 속성을 감추고 메소드로 컨트롤한다
# 3. 상속: 부모 클래스를 최대한 활용한다
# 4. 다형성: 이름도 같고 파라미터도 같은데 내부 로직이 다른 함수 = 오버라이딩
# 참고: 이름은 같은데 파라미터는 다른 함수를 만드는 것 = 오버로딩

# 객체지향 프로그램 설계의 5대 원칙
# 1. 단일 책임 원칙
# 2. 개방 폐쇄 원칙
# 3. 리스코프 치환 원칙(부모 객체를 자식 객체가 대체할 수 있어야 한다)
# 4. 인터페이스 분리 원칙
# 5. 의존 역전 원칙


# 8.
# 메타 클래스 = 파이썬에서는 클래스도 객체이다. 따라서 클래스 객체를 만드는 또 다른 클래스가 존재한다. 이를 메타 클래스라 한다
# 추상 메소드 존재 / metaclass=ABCMeta => 추상 클래스 => 부모 클래스가 자식 클래스에게 추상 메소드의 구현을 강제
# 추상 클래스로 인스턴스(객체)를 만들면 에러가 발생

# 속성: 인스턴스 변수, 클래스 변수
# 메소드: 인스턴스 메소드, 클래스 메소드, 스태틱 메소드, 매직 메소드, 추상 메소드

# 디스크립터 => 정의 + 사용법(예제 필수) 공부해 올 것

# 디스크립터란? magic method가 구현되어 있는 객체

class Descriptor(object):
    def __init__(self, age):
        self.age = age

    def __get__(self, instance, owner):
        return getattr(instance, self.age)

    def __set__(self, instance, value):
        setattr(instance, self.age, value)

    def __delete__(self, instance):
        del self.age


class Person(object):
    age = Descriptor("age")


yoolmi = Person()

yoolmi.age = 21


# 프로퍼티 => 정의 + 사용법(예제 필수) 공부해 올 것

# @property를 사용하면 getter, setter를 간단하게 구현 가능

class Person:
    def __init__(self):
        self.age = 0

    @property
    def age(self):  # getter
        return self.age

    @age.setter
    def age(self, value):  # setter
        self.age = value


yoolmi = Person()

yoolmi.age = 21

# 9.
# 모듈 패키지 from import * as 정의파트 실행파트 __name__ __main__ __init__.py
# txt 파일 쓰고 읽기, input
# csv 파일 읽고 쓰기
# 바이너리 파일 쓰고 읽기
# 컨텍스트 매니저 with
# 예외처리

# 10.
# 시퀀스 객체 => 정의 + 사용법(예제 필수) 공부해 올 것

# 리스트(수정 가능), 튜플, range, 문자열

mylist = [0, 2, 3, 4, 6, 8, 9]
mytuple = (1,)
myrange = range(len(mylist))
mystring = "yoolmi"

# 이터러블 객체 => 정의 + 사용법(예제 필수) 공부해 올 것

# 반복 가능한 객체
sh = [1, 2, 4, 5, 6]
sh.__iter__()
mixer
파이썬은
씨ㅃㅃ처럼
파이썬은
전부
다
클래스
int
3
은
객체
cpp에서
int
x = 3
sps
네



list = ['숫자든 문자든 배열이든 리스트든 클래스가 존재']
'int x = 3은 int라는 class에 3, 모든게 다 객체'
'객체, python이 느린 이유'
'코딩 커뮤니티'
'__'


# 이터레이터 => 정의 + 사용법(예제 필수) 공부해 올 것
# 값을 차례대로 꺼낼 수 있는 객체
class Count:
    def __init__(self, first, last):
        self.first = first
        self.last = last

    def __iter__(self):
        return self

    def __next__(self):
        if self.first > self.last:
            raise StopIteration
        else:
            self.first += 1
            return self.first - 1


shinhwa = Count(1998, 2021)

for i in shinhwa:
    print(i)


# 함수의 초기값 지정 => 정의 + 사용법(예제 필수) 공부해 올 것

def info(name, age, school='경희대학교'):
    print(name, age, school)


info('김민서', 21)
info('김율미', 21, '서울대학교')

# 익명함수 => 정의 + 사용법(예제 필수) 공부해 올 것
# lambda <<매개변수>>: <<리턴값>>

print((lambda x: x * 6)(5000))

f = lambda x: x * 6
print(f(5000))
메모리
절약

# 클로저 => 정의 + 사용법(예제 필수) 공부해 올 것

함수의
흐름을
저장
내
변수랑
코드를
함께
저장하고
잇고
싶을때,
그
변수는
저장하고
밖에서
접근할
수
없으니까
데이터를
비밀로
하고
싶을때


def math():
    a = 1

    def math2(x):
        return a * x

    return math2


c = math()

c(2)

직원들
월급, 사장이
원자재
값을
숨기고
싶어

우리나라
전투기, 총알
몇개


# 둘러싼 스코프의 상태값을 기억하는 함수

def start_at(x):
    def increment_by(y):
        return x + y

    return increment_by


closure1 = start_at(1)
print("closure1:", closure1(3))
closure1: 4


# 함수가 특정 상황에 처해있는걸 기억하고 싶나?

# 언패킹 방식 => 정의 + 사용법(예제 필수) 공부해 올 것

# packing은 여러개의 객체를 하나의 객체로, unpacking은 여러개의 객체를 포함하고 있는 하나의 객체를 풂

def sum(a, b, c):
    return a + b + c


numbers = [1, 2, 3]
sum(numbers)  # error

print(sum(*numbers))


# function(*args) => 정의 + 사용법(예제 필수) 공부해 올 것
# 위치 인자 패킹

def family(father, mother, *friends):
    print("아버지: ", father)
    print("어머니: ", mother)
    if friends:
        print("형제자매: ")
        for name in friends:
            print(name)


family("이진욱", '김민서', '이슬기', '이로운')


# function(**kwargs) => 정의 + 사용법(예제 필수) 공부해 올 것
# 키워드 인자 패킹

def family(father, mother, **friends):
    print("아버지: ", father)
    print("어머니: ", mother)
    if friends:
        print("형제자매: ")
        for title, name in friends.items():
            print('{} : {}'.format(title, name))


family("이진욱", '김민서', 첫째='이슬기', 둘째='이로운')

# map filter reduce => 정의 + 사용법(예제 필수) 공부해 올 것

# map

input_map = [1, 2, 3]

result = list(map(lambda i: i ** 2, input_map))

여기서
주먹밥

밖에서
복제된 주먹밥에 김가루 뿌리기 얘는 리스트로 도 되고 튜플도 되고


다시
가지고
들어오기

저기서
김가루

list_new = []  # 새로운 공장

for k in range(3):
    input_map[k] = int(input_map[k]) ** 2

좀
더
깔끔하고
간결한
표현이
가능해서
사용하지
않을까요? for문
보다는
list
comprehension이
깔끔하고, 것
보다는
map이
대체로
더
깔끔
하죠.

// for
    a = [1, 2, 3, 4, 5]
for i in range(len(a)):
    a[i] = str(a[i])

a = [1, 2, 3, 4, 5]

// list
comprehensions
a = [str(x) for x in a]

a = [1, 2, 3, 4, 5]

// map
a = list(map(str, a))

# filter

input_filter = [-2, -3, 5, 6]


def positive(x):
    return x > 0


list(filter(positive, input_filter))

# reduce

from functools import reduce

reduce(lambda x, y: x + y, [1, 2, 3, 4, 5])


# 데코레이터 => 정의 + 사용법(예제 필수) 공부해 올 것

class Math:
    @staticmethod # 클로저 개념 붙어있음
    def add(a, b):
        print(a + b)


# 제너레이터 => 정의 + 사용법(예제 필수) 공부해 올 것

def infinite_generator():
    count = 0
    while True:
        count += 1
        yield count


gen = infinite_generator()
next(gen)


# 코루틴 => 정의 + 사용법(예제 필수) 공부해 올 것

def number_coroutine():
    while True:
        x = (pass, yield(양보, 도로에 적혀 있음, 양보하라고))
        print(x)


co = number_coroutine()
next(co)

co.send(1)
co.send(2)
co.send(3)

open
run
stop
close

open

# Futures => 정의 + 사용법(예제 필수) 공부해 올 것
 # 제너+코루틴

# AsyncIO => 정의 + 사용법(예제 필수) 공부해 올 것

import asyncio


await loop.run_in_executor(None, f, par1, par2, ...)


async def main():
    print('hello')
    await asyncio.sleep(1)
    print('world')
