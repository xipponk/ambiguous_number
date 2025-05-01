import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import random

# โฟลเดอร์เก็บผลลัพธ์
output_dir = "data/augmented"
os.makedirs(output_dir, exist_ok=True)

# 🔤 ฟอนต์ลายมือที่คุณต้องเพิ่มเข้าไปเองในโฟลเดอร์ fonts/
fonts = [
    "fonts/THSarabunNew.ttf",
    "fonts/Handwritten1.ttf",
    "fonts/Handwritten2.ttf"
]

digits = "0123456789"

def generate_complex_ambiguous_digit(font_path):
    width, height = 64, 128
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # พื้นหลังขาว

    # สุ่มเลข 1–3 ตัว มาซ้อนกัน
    num_digits = random.randint(1, 3)
    chosen_digits = random.choices(digits, k=num_digits)

    for digit in chosen_digits:
        img_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype(font_path, size=random.randint(80, 110))

        x = random.randint(0, 10)
        y = random.randint(0, 20)
        color = (random.randint(150, 255), 0, 0)  # สีแดง

        draw.text((x, y), digit, font=font, fill=color)
        canvas = np.array(img_pil)

        # หมุนเล็กน้อยเพื่อเพิ่มความเบี้ยว
        M = cv2.getRotationMatrix2D((width/2, height/2), random.uniform(-10, 10), 1)
        canvas = cv2.warpAffine(canvas, M, (width, height), borderValue=(255, 255, 255))

    # Gaussian Blur
    if random.random() < 0.8:
        ksize = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), 0)

    # ใส่ Noise
    if random.random() < 0.5:
        noise = np.random.normal(0, 20, canvas.shape).astype(np.uint8)
        canvas = cv2.add(canvas, noise)

    # Motion Blur
    if random.random() < 0.3:
        k = 5
        kernel_motion_blur = np.zeros((k, k))
        kernel_motion_blur[int((k - 1)/2), :] = np.ones(k)
        kernel_motion_blur = kernel_motion_blur / k
        canvas = cv2.filter2D(canvas, -1, kernel_motion_blur)

    return canvas

# 🔁 สร้าง 100 ภาพ
for i in range(100):
    font_path = random.choice(fonts)
    img = generate_complex_ambiguous_digit(font_path)
    cv2.imwrite(os.path.join(output_dir, f"ambiguous_{i}.png"), img)