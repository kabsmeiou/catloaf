import requests

url = "http://127.0.0.1:8000/predict"
image_path = "custom_test_img/cat_loaf_script.jpg"

with open(image_path, "rb") as file:
    files = {"file": file}
    response = requests.post(url, files=files)

# Print full response
print("Status Code:", response.status_code)

# Attempt to parse JSON
try:
  print("Response Text:", response.text)
except requests.exceptions.JSONDecodeError:
  print("Failed to decode JSON. Raw response:", response.text)
