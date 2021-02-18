# class명은 대문자
class Person:
    # class는 __init__함수 클래스의 생성함수(필수 내장함수)
    # self는 Person class 자신
    # class 내장 함수의 인수에는 self가 꼭 들어감
    def __init__(self, name, age, address):
        self.name = name
        self.age = age
        self.address = address

    def greeting(self):
        print('안녕하세요, 저는 {0}입니다.'.format(self.name))
        # print(f'안녕하세요, 저는 {self.name}입니다.')

    