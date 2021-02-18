import os
import time
from urllib.request import urlretrieve
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException

# 스크롤 내려주는 함수
def scroll_down():
    scroll_count = 0

    print("ㅡ 스크롤 다운 시작 ㅡ")

    # 스크롤 위치값 얻고 last_height 에 저장
    last_height = driver.execute_script("return document.body.scrollHeight")

    # 결과 더보기 버튼을 클릭했는지 유무
    after_click = False

    while True:
        print(f"ㅡ 스크롤 횟수: {scroll_count} ㅡ")
        # 스크롤 다운
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        scroll_count += 1
        time.sleep(1)

        # 스크롤 위치값 얻고 new_height 에 저장
        new_height = driver.execute_script("return document.body.scrollHeight")

        # 스크롤이 최하단이며
        if last_height == new_height:

            # 결과 더보기 버튼을 클릭한적이 있는 경우
            if after_click is True:
                print("ㅡ 스크롤 다운 종료 ㅡ")
                break

            # 결과 더보기 버튼을 클릭한적이 없는 경우
            if after_click is False:
                if driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').is_displayed():
                    driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
                    after_click = True
                elif NoSuchElementException:
                    print("ㅡ NoSuchElementException ㅡ")
                    print("ㅡ 스크롤 다운 종료 ㅡ")
                    break

        last_height = new_height

# search_name_list = ['덤벨', '짐볼', '레더바렐', '리포머', '요가매트', '스텝퍼', '철봉', '풀업바']
search_name_list = ['고양이']
# search_name = '덤벨'

for search_name in search_name_list :
    url = f'https://www.google.com/search?q={quote_plus(search_name)}&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiu0MK3nO3uAhUXZt4KHTL7DtwQ_AUoAXoECAUQAw&biw=1920&bih=937'

    driver = webdriver.Chrome('./project/chromedriver_win32/chromedriver.exe')
    driver.get(url)

    # for i in range(500):
    #     driver.execute_script("window.scrollBy(0,10000)")
    scroll_down()

    html = driver.page_source
    soup = BeautifulSoup(html)

    img = soup.select('.rg_i.Q4LuWd')
    n = 1
    imgurl = []
    img_folder_path = './project/data/img/'
    img_folder_name = search_name.replace(' ', '')
    after_click = False     # 결과 더보기 버튼을 클릭했는지 유무

    if not os.path.isdir(img_folder_path+img_folder_name): 
            os.mkdir(img_folder_path + img_folder_name + "/")

    for i in img:
        try:
            imgurl.append(i.attrs["src"])
        except KeyError:
            imgurl.append(i.attrs["data-src"])

    for i in imgurl:
        # urlretrieve(i, './project/data/img/' + str(search_name) + '/' + str(search_name)+ str(n) + '.jpg')
        urlretrieve(i, f'{img_folder_path}{img_folder_name}/{img_folder_name}{n}.jpg')
        n += 1 
        print(imgurl)
        # if(n == 400):
        #     break

    driver.close()



'''
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import time
import re 
import shutil
import os 

def save_img(directory,driver,count):
    html = driver.page_source
    soup = BeautifulSoup(html,'lxml')
    url = soup.select("div.image > img")[0].get('src')

    resp=requests.get(url,stream=True)
    filename = directory+'/test_{}.jpg'.format(str(count))
    local_file = open(filename,'wb')
    resp.raw.decode_content=True
    shutil.copyfileobj(resp.raw,local_file)
    return 

def next_to(driver):
    try:
        driver.find_element_by_xpath("//*[@id=\"main_pack\"]/section/div[2]/div[2]/div/div[1]/div[1]/div[2]/a[2]/i").click()
    except:
        try:
            driver.find_element_by_xpath("//*[@id=\"main_pack\"]/section/div/div[2]/div/div[1]/div[1]/div[2]/a[2]").click()
        except:
            driver.execute_script('window.scrollTo(0, 1);')
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    return

keyword_list=['덤벨', '짐볼', '레더바렐', '리포머', '요가매트', '스텝퍼', '철봉', '풀업바', '기구필라테스 캐딜락']
picture_num=10

for keyword in keyword_list:
    img_folder_path = './project/data/img/'

    driver = webdriver.Chrome('./project/chromedriver_win32/chromedriver.exe')
    # driver.get('https://search.naver.com/search.naver?where=image&sm=tab_jum&query={}'.format(keyword))
    driver.get()
    time.sleep(3)
    driver.find_element_by_xpath("//*[@id=\"main_pack\"]/section/div[2]/div[1]/div[1]/div[1]/div/div[1]/a").click()
    html = driver.page_source
    soup = BeautifulSoup(html,'lxml')
    url = soup.select("div.image > img")[0].get('src')
    count=0
    scroll_count = 0


    if not os.path.isdir(img_folder_path+keyword): 
        os.mkdir(img_folder_path+keyword+"/")

    while count<=picture_num-1:
        save_img(img_folder_path+keyword,driver,count)
        next_to(driver)
        count+=1
        scroll_count+=1
        if scroll_count>=12:
            print(count)
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            scroll_count=0
            time.sleep(3) 
        time.sleep(0.5)

         '''


''' 
from selenium import webdriver
from bs4 import BeautifulSoup as soups

def search_selenium(search_name, search_path, search_limit) :
        search_url = "https://www.google.com/search?q=" + str(search_name) + "&hl=ko&tbm=isch"
        
        browser = webdriver.Chrome('./project/chromedriver_win32/chromedriver.exe')
        browser.get(search_url)
        
        image_count = len(browser.find_elements_by_tag_name("img"))
        
        print("로드된 이미지 개수 : ", image_count)

        browser.implicitly_wait(2)

        for i in range( image_count ) :
            image = browser.find_elements_by_tag_name("img")[i]
            image.screenshot('./project/data/img/'+ str(search_name) + '/' + str(i) + ".jpg")

        browser.close()

search_name_list = ['덤벨', '짐볼', '레더바렐', '리포머', '요가매트', '스텝퍼', '철봉', '풀업바']
search_limit = 10

for search_name in search_name_list:

    search_path = "Your Path"
    search_selenium(search_name, search_path, search_limit)


'''

