import os
import base64
from PIL import Image
import io

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        img = Image.open(img_file)
        max_side = 640
        scale = max_side / max(img.width, img.height)
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=70)
        buffer.seek(0)
        img_file = buffer
        return base64.b64encode(img_file.read()).decode("utf-8")
    
table = "image_data.csv"
folder = "SemArt/Images"

# Check if the file exists and write headers if it doesn't
if not os.path.exists(table):
    with open(table, "w") as file:
        file.write("image_name,base_64\n")

for image in os.listdir(folder):
    image_path = os.path.join(folder, image)
    base64_image = convert_image_to_base64(image_path)
    with open(table, "a") as file:
        file.write(f"{image},{base64_image}\n")