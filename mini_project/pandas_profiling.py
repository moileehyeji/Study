import pandas as pd
import pandas_profiling

data = pd.read_csv('./dacon/computer2/data/dirty_mnist_2nd_answer.csv',encoding='latin1')

# print(data[:5])

# 2. 리포트 생성하기
pr=data.profile_report() # 프로파일링 결과 리포트를 pr에 저장
# data.profile_report() # 바로 결과 보기

# pr.to_file('./pr_report.html') # pr_report.html 파일로 저장

# 3. 리포트 살펴보기
print(pr)

# AttributeError: 'DataFrame' object has no attribute 'profile_report'