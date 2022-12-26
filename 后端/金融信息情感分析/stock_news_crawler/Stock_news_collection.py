# -*- coding: utf-8 -*-
"""
Welcome to Noah's Ark
"""
import json
import requests
from requests.exceptions import ConnectionError
from requests_html import HTMLSession

class StockNewsCollection():
    '''
    股票资讯爬虫
    源网页：https://www.eastmoney.com/
    Demo
    ----------
        #注册
        snc = StockNewsCollection()
        ----------
        #获取{'关键词':'芳源'}相关资讯100条
        err, maxn, result = snc.crawling('芳源', 100)
    '''
    def __init__(self):
        self.url = 'https://searchapi.eastmoney.com/bussiness/Web/GetCMSSearchList?cb=?'
        self.headers = {
                        'Accept': '*/*',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6',
                        'Connection': 'keep-alive',
                        'Cookie': 'qgqp_b_id=17ef720316f256a000a9999999bdf555; em_hq_fls=js; intellpositionL=1114.7px; intellpositionT=1261.61px; emshistory=%5B%22noe%22%5D; HAList=a-sh-605588-N%u51A0%u77F3; st_si=74431893018251; st_asi=delete; st_pvi=43061953490523; st_sp=2021-08-08%2021%3A50%3A36; st_inirUrl=https%3A%2F%2Fwww.baidu.com%2Flink; st_sn=20; st_psi=20221231174613342-111000300841-2576687711',
                        'Host': 'searchapi.eastmoney.com',
                        'Referer': 'https://www.eastmoney.com/',
                        'sec-ch-ua': '"Chromium";v="92", " Not A;Brand";v="99", "Microsoft Edge";v="92"',
                        'sec-ch-ua-mobile': '?0',
                        'Sec-Fetch-Dest': 'script',
                        'Sec-Fetch-Mode': 'no-cors',
                        'Sec-Fetch-Site': 'same-site',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36 Edg/92.0.902.73'
                        }
        self.data = {
                    'keyword': '',
                    'type': 8196,
                    'pageindex': 1,
                    'pagesize': 10,
                    'name': 'web'
                    }
    def __str__(self):
        return 'https://www.eastmoney.com/'
    def crawling(self, keyword, nums=10):
        '''
        爬虫主函数
        StockNewsCollection.crawling(keyword, nums=10)
        作用：爬取搜索内容的相关资讯
        Parameters
        ----------
            keyword(str):关键词，即搜索内容
            nums(int):需要爬取的资讯数量，单位(条)
        Return
        ----------
            Form: err, total, result
            err(int):错误码[0:999]
                #0   no error
                #100 keyword==''
                #200 搜索结果为空
                #404 网络连接异常
                #406 获取链接失败
                #999 未知异常(Impossible Error)
            total(int):该关键词最大可爬取数量
            result(list):爬取结果，一般形式 [[时间, 链接, 标题, 文章], ..., [时间, 链接, 标题, 文章]]
        '''
        if type(keyword) != str:
            keyword = str(keyword)
        if keyword == '':
            return 100, 0, [] #空搜索
        if type(nums) == float:
            nums = int(nums)
        if type(nums) != int or nums < 1:
            nums = 1
        self.data['keyword'] = keyword
        self.data['pagesize'] = nums
        err, total, links = self.__get_links()
        if err:
            return err, 0, []
        #print(links)
        result = []
        for lk in links:
            tt = self.__crawl_text(lk[1])
            if type(tt) != list:
                continue
            result.append(lk + tt)
        return 0, total, result
    def __get_links(self): #获取资讯网页链接
        try:
            response = requests.get(self.url, headers=self.headers, data=self.data)
            resp = json.loads(response.text[2:-1])
        except ConnectionError:
            return 404, 0, [] #网络异常
        except:
            return 999, 0, [] #未知异常
        if not resp['IsSuccess']:
            return 406, 0, [] #获取链接失败
        if resp['Data'] == None:
            return 200, 0, []
        return 0, resp['TotalCount'], [[uni['Art_CreateTime'], uni['Art_UniqueUrl']] for uni in resp['Data']]
    def __crawl_text(self, url_t): #资讯文本爬取
        try:
            temp_page = HTMLSession().get(url_t).html
            #temp_page.render()
            temp_res = []
            temp_res.append(temp_page.find("div.title")[0].text)
            inf = temp_page.find("div.txtinfos")[0].find("p")
            text = ''
            for info in inf:
                if info.attrs.get('class') == None:
                    text += info.text + '\n'
            temp_res.append(text)
            #print(temp_res[0])
            return temp_res
        except ConnectionError:
            return 404
        except:
            return 999
        
if __name__ == '__main__':
    snc = StockNewsCollection()
    err, maxn, res = snc.crawling(688148, 2)
    #688148
    print(err, res)