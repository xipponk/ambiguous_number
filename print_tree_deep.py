import os

def print_tree(start_path, prefix=''):
    items = sorted(os.listdir(start_path))
    for i, name in enumerate(items):
        path = os.path.join(start_path, name)
        is_last = (i == len(items) - 1)
        connector = '└── ' if is_last else '├── '
        print(prefix + connector + name)
        if os.path.isdir(path):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            print_tree(path, new_prefix)

# target directory
target_directory = r"C:\Users\tuul.tri\Projects\ambiguous_number"

if __name__ == "__main__":
    print(f"Directory structure for: {target_directory}")
    print_tree(target_directory)