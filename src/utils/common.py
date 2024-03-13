import os


def extract_image_path_list(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        # root直下の画像ファイルを取得
        for file in files:
            if any(
                file.lower().endswith(ext)
                for ext in [".jpg", ".jpeg", ".JPG", ".png", ".PNG"]
            ):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                image_files.append(relative_path)
    return image_files
