import os

font_dir = "fonts"
font_files = [
    f'"{os.path.join(font_dir, file).replace("\\", "/")}"'
    for file in os.listdir(font_dir)
    if file.endswith(".ttf")
]

print("fonts = [")
print(",\n".join(font_files))
print("]")