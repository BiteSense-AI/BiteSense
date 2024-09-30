import os
import requests
from PIL import Image
from io import BytesIO

API_KEY = "AIzaSyBPn6hnagWwcyZ-YYb9ZYmGWPxgrSE1oPA"
CX = "b28bc8704770e44ff"

search_url = "https://www.googleapis.com/customsearch/v1"

downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
output_dir = os.path.join(downloads_folder, "mosquito_bite_images")
os.makedirs(output_dir, exist_ok=True)

def download_images(query, num_images):
    params = {
        "q": query,
        "cx": CX,
        "key": API_KEY,
        "searchType": "image",
        "num": 10,
        "imgSize": "medium"
    }

    downloaded = 0
    start_index = 1
    while downloaded < num_images:
        params["start"] = start_index
        response = requests.get(search_url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break

        search_results = response.json()

        if "items" not in search_results:
            print("No more images found.")
            break

        for i, item in enumerate(search_results["items"]):
            try:
                img_url = item["link"]
                img_data = requests.get(img_url)
                img_data.raise_for_status()
                img = Image.open(BytesIO(img_data.content))
                img = img.convert('RGB')
                img.save(os.path.join(output_dir, f"mosquito_bite_{downloaded + 1}.jpg"))
                downloaded += 1
                print(f"Downloaded image {downloaded}: {img_url}")
                if downloaded >= num_images:
                    break
            except Exception as e:
                print(f"Failed to download image {downloaded + 1}: {e}")

        start_index += 10

search_query = "mosquito bite on skin"
download_images(search_query, 500)

print(f"Downloaded mosquito bite images to {output_dir}")
