import requests

# Replace with your actual API key and CSE ID
API_KEY = "AIzaSyAG2Sxh8oUSNVWYuxoZ8ZV8TcnHOd81apM"
CSE_ID = "713fd10114bfe4c8f"

query = "Python programming"  # Any test query
search_type = "image"         # Use "image" for image search, omit for web search

url = "https://www.googleapis.com/customsearch/v1"
params = {
    "key": API_KEY,
    "cx": CSE_ID,
    "q": query,
    "searchType": search_type,  # ✅ must be camelCase
    "num": 2                    # optional: number of results (max 10)
}

response = requests.get(url, params=params)

# Check if request succeeded
if response.status_code == 200:
    data = response.json()
    items = data.get("items", [])
    if not items:
        print("No results returned. Check your CSE settings!")
    else:
        print(f"Success! Returned {len(items)} results:")
        for i, item in enumerate(items, 1):
            print(f"{i}. {item.get('title')} - {item.get('link')}")
else:
    print(f"Error: {response.status_code} - {response.text}")
