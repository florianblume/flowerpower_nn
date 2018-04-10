def get_files_at_path_of_extensions(images_path, extensions):
    import os
    return [fn for fn in os.listdir(images_path) if any(fn.endswith(ext) for ext in extensions)]