import requests as requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
from tqdm import tqdm
from time import sleep


path = '/Users/yuwenhan/Desktop/python/images'
os.makedirs(path, exist_ok=True)
ct=0
for i in range(1, 44):
    r = requests.get("https://pixabay.com/api/", params={
        "key": "31867904-f6111a2529031a07d461ca2be",
        "q": "nature scene",
        "image_type": "photo",
        "category": "nature",
        "min_width": 512,
        "min_height": 512,
        "page": i,  
        "per_page": 10 # 最多 200
    })

    print(r.headers)
    urls = [x['webformatURL'] for x in r.json()['hits']]
    
    for url in tqdm(urls):
        r = requests.get(url)
        
        with open(os.path.join(path, f"wen_{ct}.jpg"), 'wb') as f:
            for chunk in r:
                f.write(chunk)
            ct+=1

    print(f"[INFO] page {i} finished!")
    # sleep(3)
