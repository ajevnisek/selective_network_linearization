import os
from PIL import Image


def create_gif_from_images(directory_path, target_gif_path):
    images = []

    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    # Iterate through the files in the directory
    for filename in sorted(os.listdir(directory_path)):
        filepath = os.path.join(directory_path, filename)
        try:
            # Open the image file
            img = Image.open(filepath)
            # Append image to the list
            images.append(img)
        except IOError:
            print(f"Unable to load image file '{filepath}'")

    # Save as a GIF
    if images:
        try:
            # Save the images as a GIF
            images[0].save(target_gif_path,
                           save_all=True,
                           append_images=images[1:],
                           duration=200,  # Set the duration (in milliseconds) between each frame
                           loop=0)  # Set loop to 0 for an infinite loop, or a positive integer for a finite loop count
            print(f"Generated GIF saved at '{target_gif_path}'")
        except Exception as e:
            print(f"Failed to save GIF: {e}")
    else:
        print("No images found in the directory.")