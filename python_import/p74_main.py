import p73_class as p73

# 인스턴스 생성
malddong = p73.Person('이혜지', 27, '수원시')

malddong.greeting() #안녕하세요, 저는 이혜지입니다.
# greeting함수 인수에 self 지우면 에러
# TypeError: greeting() takes 0 positional arguments but 1 was given

# pip install로 설치가 가능하도록 배포하는 방법은?