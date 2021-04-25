import pandas as pd

submit_path = '../data/kaggle/titanic/submission_test1.csv'
submit = pd.read_csv(submit_path, sep=',').astype('float32')

submit['PassengerId'] = submit['PassengerId'].astype('int32')
submit['Survived'] = submit['Survived'].astype('int32')

submit.to_csv('../data/kaggle/titanic/submission_test1-1.csv', index=False)


print(submit.info())
print(submit)