from PIL import Image
import os

image_path = 'D:/hacker/energivanu/Energivanu2/ChatGPT Image Jun 29, 2026, 03_52_18 PM.png'
output_dir = 'D:/hacker/energivanu/Energivanu2/magazine/assets'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open image
img = Image.open(image_path)
width, height = img.size

# The grid is 3 columns and 4 rows (3 * 4 = 12 slides)
cols = 3
rows = 4

slide_w = width / cols
slide_h = height / rows

print(f"Image dimensions: {width}x{height}")
print(f"Cropping grid: {cols} columns x {rows} rows. Each slide: {slide_w}x{slide_h}")

slide_index = 1
for r in range(rows):
    for c in range(cols):
        left = c * slide_w
        top = r * slide_h
        right = left + slide_w
        bottom = top + slide_h
        
        # Crop
        slide = img.crop((left, top, right, bottom))
        
        # Save
        filename = f"slide_{slide_index:02d}.png"
        filepath = os.path.join(output_dir, filename)
        slide.save(filepath, "PNG")
        print(f"Saved: {filepath}")
        slide_index += 1

print("All 12 slides cropped successfully.")
