import os

def print_tree_one_level(path):
    print(f"Directory structure (1 level) for: {path}")
    try:
        items = sorted(os.listdir(path))
    except FileNotFoundError:
        print(f"[Error] Path not found: {path}")
        return

    for name in items:
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            print(f"📁 {name}/")
        else:
            print(f"📄 {name}")

# เปลี่ยน path ที่นี่
target_directory = r"C:\Users\tuul.tri\Projects\ambiguous_number"

if __name__ == "__main__":
    print_tree_one_level(target_directory)