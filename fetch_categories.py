import requests

# URL API
# url = "https://e34b199081e2.ngrok-free.app/api/v1/categories"
url = "https://830e8519fa0b.ngrok-free.app/api/v1/categories"

# Token của bạn
# token = "eyJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJ0ZXN0LmNvbSIsInN1YiI6ImRlbW9AZ21haWwuY29tIiwiZXhwIjoxNzY0MDg0MjI3LCJpYXQiOjE3NjQwNjQyMjcsImp0aSI6IjIzZGQ3YWZiLThlOTUtNDVkMi04YjQ0LWI2ZWJhYzk5OGQ0MCIsInNjb3BlIjoiUk9MRV9VU0VSIn0.w4hv3OeKCYtax6pRQ-nVKkdLb7dYmgWQ5b3x5JiC4vBHR765h7G73jFxP3zphYZwwY3kbDnxtxfLRRK_1JqWLA"

# Header Authorization
# headers = {
#     "Authorization": f"Bearer {token}"
# }

# Gọi API
response = requests.get(url)
data = response.json()

# Lấy danh sách category
categories = [item['categoryname'] for item in data['result']]
print(categories)
