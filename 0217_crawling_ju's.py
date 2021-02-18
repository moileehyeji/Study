from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import urllib.request
import os

# 'gym ball', 'ladder barrel', dumbbell'
# ,'yogamat',, 'running machine', 'chining dipping'     
search_name_list = ['running machine', 'dumbbell', 'rowingmachine']
img_folder_path = './project/data/img3'

for search_name in search_name_list :
    img_folder_name = search_name.replace(' ', '')
    
    # select webbrowser (chrome)
    driver = webdriver.Chrome('./project/chromedriver_win32/chromedriver.exe')

    # link to address
    driver.get('https://www.google.co.kr/imghp?hl=ko&tab=ri&ogbl')

    # fine specified elements
    elem=driver.find_element_by_name('q')

    # input keys & enter
    elem.send_keys(search_name)
    elem.send_keys(Keys.RETURN)

    # scroll web page
    SCROLL_PAUSE_TIME=1
    last_height=driver.execute_script('return document.body.scrollHeight')
    # 스크롤 높이를 java Script 로 찾아서 last_height 란 변수에 저장 시킴

    while True: # 무한 반복
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        # 스크롤을 끝까지 내린다

        time.sleep(SCROLL_PAUSE_TIME) # 스크롤이 끝나면 1초동안 기다림

        new_height=driver.execute_script('return document.body.scrollHeight')
        if new_height==last_height:
            try:
                driver.find_element_by_css_selector('.mye4qd').click()
                # 결과 더보기 버튼 클릭
            except:
                break
        last_height=new_height

    # select & click image in webbrowser 
    images=driver.find_elements_by_css_selector('.rg_i.Q4LuWd')
    count=1

    if not os.path.isdir(f'{img_folder_path}/{img_folder_name}2'): 
            os.mkdir(f'{img_folder_path}/{img_folder_name}2/')

    for image in images:
        try:
            image.click() # 인터넷 상의 이미지 클릭
            time.sleep(3) # 이미지 로드 시간을 위해 지연시간 추가
            imgUrl=driver.find_element_by_xpath('/html/body/div[2]/c-wiz/div[3]/div[2]/div[3]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div/div[2]/a/img').get_attribute('src') # 저장할 이미지 경로
            # if imgUrl==driver.find_element_by_link_text('https://images.costco-static.com/ImageDelivery/imageService?profileId=12026540&itemId=1462223-847&recipeName=680'):
            #     print('tq')
            opener=urllib.request.build_opener()
            opener.addheaders=[('User-Agent','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(imgUrl, f'{img_folder_path}/{img_folder_name}2/{img_folder_name}{count}.jpg') # 이미지 저장
            count=count+1 # 이미지 파일 이름을 순서대로 맞추기 위해 증가시킴
            
            time.sleep(5) # 저장 후 페이지 로드 할 시간을 위해 지연시간 추가
            '''
            Forbidden 이 뜨면 위의 코드를 추가한다
            python 으로 제어 되는 브라우저를 봇으로 인식하는 경우,
            위의 header 를 추가해주면 해당 문제를 벗어날 수 있다.
            '''

        except:
            pass

    driver.close() # 웹페이지 종료