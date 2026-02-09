from rembg import remove
from PIL import Image
import os

input_path = "/Users/yoyocubano/.gemini/antigravity/brain/aefc49bb-149d-46db-98ae-d38febf6939d/media__1770656128568.jpg"
output_path = "/Users/yoyocubano/Documents/AIGOTAJOB/ui/assets/welux_eagle_isolated.png"

# Ensure assets directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

print(f"Opening image: {input_path}")
input_image = Image.open(input_path)

print("Removing background...")
output_image = remove(input_image)

print(f"Saving isolated image to: {output_path}")
output_image.save(output_path)
print("Done!")
