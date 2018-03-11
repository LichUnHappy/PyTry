import re
import time

import requests

# headers = {
#     'User-Agent':'Mozilla/5.0'
# }
f = open('test.txt','a+')

def get_info(url):
    res = requests.get(url)
    if res.status_code == 200:
        contents = re.findall('<p>(.*?)</p>', res.content.decode('utf-8'),re.S)
        for content in contents:
            f.write(content+'\n')
    else:
        pass
if __name__ == '__main__':
    urls = ['http://www.doupoxs.com/doupocangqiong/{}.html'.format
    (str(i)) for i in range(2, 1665)]
    for url in urls:
        get_info(url)
        time.sleep(1)
f.close()
print('ok')
