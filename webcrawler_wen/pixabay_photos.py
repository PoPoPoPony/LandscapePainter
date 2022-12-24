import requests as requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from tqdm import tqdm
from time import sleep

path = '/Users/yuwenhan/Desktop/python/images'
os.makedirs(path, exist_ok=True)

def Pixabay_keyword(page_s, page_end, usekey):
    ct=0
    for i in range(page_s, page_end):
        r = requests.get("https://pixabay.com/api/", params={
            "key": "31867904-f6111a2529031a07d461ca2be",
            "q": usekey,
            "image_type": "photo",
            "category": "nature",
            "min_width": 512,
            "min_height": 512,
            "page": i,  
            "per_page": 200 # 最多 200
        })

        print(r.headers)
        urls = [x['webformatURL'] for x in r.json()['hits']]
        
        for url in tqdm(urls):
            r = requests.get(url)
            
            with open(os.path.join(path, f"pixabay_wen_{ct}.jpg"), 'wb') as f:
                for chunk in r:
                    f.write(chunk)
                ct+=1

        print(f"[INFO] page {i} finished!")
        sleep(3)

Pixabay_keyword(1, 45, 'nature scene')