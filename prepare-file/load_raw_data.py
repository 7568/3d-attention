import random
import time

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import sys
import os
sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
from library import common as cm

"""
get 7 days fixing depository-institutions repo rate from https://www.chinabond.com.cn/
"""


def do_mutil_process(param):
    pages = param['pages']
    # time.sleep(pages[0])
    return get_page_list_info(pages)
#1168135

def mutil_process():
    cpu_num = cpu_count() - 16
    pages = np.arange(1, 161)
    # pages = np.array([19,20])
    data_chunks = cm.chunks_np(pages, cpu_num)
    param = []
    for data_chunk in data_chunks:
        ARGS_ = dict(pages=data_chunk)
        param.append(ARGS_)
    with Pool(cpu_num) as p:
        r = p.map(do_mutil_process, param)
        p.close()
        p.join()
    print('run done!')
    new_df = pd.DataFrame(columns=['date', 'rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21'])
    for _r in r:
        new_df = new_df.append(_r)
    new_df.to_csv('date_rate.csv', index=False)


def get_page_list_info(pages):
    re = []
    dd = pd.DataFrame(columns=['date', 'rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21'])
    count = 0
    for i in pages:
        while True:
            try:
                url = f'https://www.chinabond.com.cn/cb/cn/zzsj/sjtj/jsrb/list_{i}.shtml'  # 要爬取的网页 URL
                user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.{np.random.randint(1, 100)} (KHTML, like Gecko) Chrome/111.{np.random.randint(1, 50)}.0.0 Safari/537.{np.random.randint(1, 100)}"
                headers = {
                    'User-Agent': user_agent}
                time.sleep(1)
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.content, 'html.parser')

                data = soup.select('div.rightContentBox div.rightListContent')
                if len(data) > 0:
                    break
                time.sleep(5)
            except:
                print(f'url fail : {url}')

        for d in data:
            href = d.find('a').get('href')
            print(f'i:{i} , href : {href}')
            re.append(href)

            while True:
                # print(f'get_page_info(href) : {get_page_info(href)}')
                time.sleep(1)
                row_info = get_page_info(href)
                print(f'len(row_info) : {len(row_info)}')
                if len(row_info) > 0:
                    break
                time.sleep(5)
            # print(row_info)
            dd.loc[count] = row_info
            count += 1
        print(f'save date_rate_{i}.csv and size is {dd.size}')
        dd.to_csv(f'date_rate_{i}.csv', index=False)
        return dd

    # print(data)


def get_page_info(href):
    date_str = href.split('jsrb')[1].split('/')[1]
    date_str = f'{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}'
    print(f'date_str : {date_str}')
    user_agent = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.{np.random.randint(1, 100)} (KHTML, like Gecko) Chrome/111.{np.random.randint(1, 50)}.0.0 Safari/537.{np.random.randint(1, 100)}"
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.96 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36"
    ]
    headers = {
        'User-Agent': random.choice(user_agents),
        "Content-Type": "text/plain;charset=UTF-8",
    'Accept-Encoding': 'utf-8'}


    response_2 = requests.get(f'https://www.chinabond.com.cn/{href}', headers=headers)
    # html_2 = response_2.content
    soup_2 = BeautifulSoup(response_2.content, 'html.parser')
    row_info = [date_str]
    for day_i in [1, 2, 3, 4, 5, 6]:
        try:
            tag='p'
            str = '当日利率'
            if soup_2.find(tag,string=str) is None:
                tag='span'
            if soup_2.find(tag,string=str) is None:
                str='当日利率 '
            if soup_2.find(tag, string=str) is None:
                tag = 'p'
            if soup_2.find(tag, string=str) is None:
                if len(soup_2.select('table')[2].select('tbody tr')[2].select('td')[day_i].select('p span')) > 2:
                    # print(1)
                    row_info.append(
                        soup_2.select('table')[2].select('tbody tr')[2].select('td')[day_i].select('p span')[2].text)
                else:
                    # print(2)
                    row_info.append(
                        soup_2.select('table')[2].select('tbody tr')[2].select('td')[day_i].select('p span')[0].text)
            else:
                if len(soup_2.find(tag, string=str).find_parent('tr').select('td')[day_i].select('p span')) > 2:
                    # print(3)
                    row_info.append(
                        soup_2.find(tag, string=str).find_parent('tr').select('td')[day_i].select('p span')[2].text)
                else:
                    # print(4)
                    row_info.append(
                        soup_2.find(tag, string=str).find_parent('tr').select('td')[day_i].select('p span')[0].text)
        except Exception as e:
            print(f'error link is https://www.chinabond.com.cn/{href}')
            # print(html_2)
            # # html_file = open(f'{day_i}.html','w')
            # with open(f'{day_i}.html','w') as html_file:
            #     html_file.write(html_2)
            # print(e)
            # raise 1
            return []
    # print(row_info)
    return row_info



# np.savetxt('link.csv', np.array(re), delimiter=',')
def single_process():
    dd = pd.DataFrame(columns=['date', 'rate_1', 'rate_2', 'rate_3', 'rate_7', 'rate_14', 'rate_21'])
    pages = np.arange(1, 161)
    dds = get_page_list_info(pages)
    for d in dds:
        dd.append(d)
    dd.to_csv('date_rate.csv', index=False)


if __name__ == '__main__':
    mutil_process()


    # single_process()
    # https://www.chinabond.com.cn//cb/cn/zzsj/sjtj/jsrb/20220524/160340947.shtml
    # get_page_info('/cb/cn/zzsj/sjtj/jsrb/20220524/160340947.shtml')
