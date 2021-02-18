
# 문제점 :  같은 폴더 내의 모듈만 호출할수 있음
# --> sklearn는 어떻게 호출할 수 있었을까?
# --> Anaconda 설치할때 환경변수 체크한것이 'C:\\Users\\ai\\Anaconda3' 하위 폴더 모든 경로의 모듈을 호출할 수 잇도록 설정한 것

import sys

print(sys.path)

# 제어판\모든 제어판 항목\시스템\고급 시스템 설정\고급\환경변수\Path
# 내부를 보면 C:\Users\ai\Anaconda3
# 시스템 변수\CUDA_PATH확인하면 현재 실행중인 CUDA 버전확인이 가능하다.