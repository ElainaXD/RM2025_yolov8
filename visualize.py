import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def draw_boxes_from_file(file_path, img_path):
    # Load the image
    img = plt.imread(img_path)
    img_height, img_width = img.shape[0], img.shape[1]
    
    with open(file_path, 'r') as f:
        data = [list(map(float, line.strip().split())) for line in f.readlines()]
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)  # Display the loaded image

    for box in data:
        _, x, y, w, h = box
        x_center = x * img_width
        y_center = y * img_height
        width = w * img_width
        height = h * img_height

        # Convert center x,y to top-left x,y
        x_top_left = x_center - (width / 2)
        y_top_left = y_center - (height / 2)

        # Draw the bounding box
        rect = patches.Rectangle((x_top_left, y_top_left), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        print("boundary box created")

    # Save the visualization as an image file
    output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_visualized.png"
    output_path = os.path.join(os.path.dirname(file_path), output_filename)
    fig.savefig(output_path)
    plt.close(fig)  # Close the figure to free up memory

def main():
    txt_directory = r"/app/test"
    jpg_directory = r"/app/test" # Update with the path to your JPG images folder

    for filename in os.listdir(txt_directory):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(txt_directory, filename)
            # Assume the jpg has the same name as the txt file, just with a different extension
            jpg_file_path = os.path.join(jpg_directory, os.path.splitext(filename)[0] + ".jpg")
            if os.path.exists(jpg_file_path):  # Ensure the jpg file exists
                draw_boxes_from_file(txt_file_path, jpg_file_path)
    

if __name__ == '__main__':
    main()
