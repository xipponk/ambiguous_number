import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import random

output_dir = "data/augmented"
os.makedirs(output_dir, exist_ok=True)

fonts = sorted([
    "fonts/Kart-Thai raimue DEMO.ttf",
    "fonts/Kart-Thai raimue2 DEMO.ttf",
    "fonts/nura-kaopan-thin.ttf",
    "fonts/nura-niwow-thin.ttf",
    "fonts/SOV_PyaNakh.ttf"
])

digits = "0123456789"

def generate_gan_ready_digit(font_path):
    width, height = 64, 64
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255
    chosen_digits = random.choices(digits, k=3)
    base_y_positions = [0, 20, 40]

    for i, digit in enumerate(chosen_digits):
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)

        try:
            font_size = random.randint(30, 45)
            font = ImageFont.truetype(font_path, size=font_size)
        except OSError as e:
            print(f"[Warning] Could not load font: {font_path} → {e}")
            continue

        x = random.randint(5, 15)
        y = base_y_positions[i] + random.randint(-2, 2)
        if y + font_size > height:
            y = height - font_size - 1

        color = (random.randint(150, 255), 0, 0)
        draw.text((x, y), digit, font=font, fill=color)
        canvas = np.array(img_pil)

        M = cv2.getRotationMatrix2D((width / 2, height / 2), random.uniform(-10, 10), 1)
        canvas = cv2.warpAffine(canvas, M, (width, height), borderValue=(255, 255, 255))

    if random.random() < 0.8:
        ksize = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), 0)

    return canvas

# Generate as many images as you want
num_images = 50000
for i in range(num_images):
    font_path = random.choice(fonts)
    img = generate_gan_ready_digit(font_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filename = f"ambiguous_{i:05}.png"
    cv2.imwrite(os.path.join(output_dir, filename), img_gray)