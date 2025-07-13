# python -m serpengine.serpapi

from serpapi import GoogleSearch


params = {
  "api_key": "fd5719239320281ab1165b978da8b5df70d55d2ce22f41f1e2031cbdd3b3b8e5",
  "engine": "google",
  "q": "MOTORCU HAYRİ MOTORSİKLET VE BİSİKLET SANAYİ VE TİCARET LİMİTED ŞİRKETİ",
  "location": "Austin, Texas, United States",
  "google_domain": "google.com",
  "gl": "us",
  "hl": "en"
}

search = GoogleSearch(params)
results = search.get_dict()

print(results)