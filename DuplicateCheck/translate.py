import random
import requests
import hashlib

appid = '20210930000961232'
secretKey = 'o1dteOfDa6ZBxC7hKJtl'


def get_md5(string):
    hl = hashlib.md5()
    hl.update(string.encode('utf-8'))
    return hl.hexdigest()


def e2z(en_str):
    api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    sign = get_md5(appid + en_str + str(salt) + secretKey)
    api_data = {
        'q': en_str,
        'from': 'en',
        'to': 'zh',
        'appid': appid,
        'salt': salt,
        'sign': sign
    }
    req_get = requests.get(api_url, api_data)
    result = req_get.json()
    return result['trans_result']


def z2e(zh_str):
    api_url = 'https://fanyi-api.baidu.com/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    sign = get_md5(appid + zh_str + str(salt) + secretKey)
    api_data = {
        'q': zh_str,
        'from': 'zh',
        'to': 'en',
        'appid': appid,
        'salt': salt,
        'sign': sign
    }
    req_get = requests.get(api_url, api_data)
    result = req_get.json()
    return result['trans_result']


def rewrite(sentence):
    return e2z(z2e(sentence)[0]['dst'])[0]['dst']

if __name__ == "__main__":
    print(z2e("你好世界"))
