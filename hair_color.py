import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import os
import argparse

# Argument parser to take command-line inputs
def parse_arguments():
    parser = argparse.ArgumentParser(description='Apply hair color to an image using segmentation.')
    parser.add_argument('--model', type=str, required=True, help='Path to the TFLite hair segmentation model.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--color', type=str, required=True, choices=['pink', 'violet', 'maroon', 'apple_green', 'orange'],
                        help='Choose a hair color: pink, violet, maroon, apple_green, orange.')
    parser.add_argument('--output_dir', type=str, default='/content/makeup_results', help='Directory to save the output images.')
    
    return parser.parse_args()

# Define hair colors in RGB
hair_colors_rgb = {
    'pink': np.array([255, 192, 203], dtype=np.uint8),
    'violet': np.array([238, 130, 238], dtype=np.uint8),
    'maroon': np.array([128, 0, 0], dtype=np.uint8),
    'apple_green': np.array([144, 238, 144], dtype=np.uint8),
    'orange': np.array([255, 165, 0], dtype=np.uint8)
}

# Main function
def apply_hair_color(args):
    # Define paths
    model_path = args.model
    image_path = args.image
    selected_color = args.color
    output_dir = args.output_dir

    # Initialize MediaPipe Image Segmenter for hair segmentation
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create an image segmenter instance
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True
    )

    with ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image.create_from_file(image_path)
        segmented_masks = segmenter.segment(mp_image)

    # Extract the mask
    category_mask = segmented_masks.category_mask
    mask = category_mask.numpy_view()

    # Load the original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to LAB color space
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)

    # Extract L channel
    L_channel, A_channel, B_channel = cv2.split(image_lab)

    # Select the desired hair color in RGB
    hair_color_rgb = hair_colors_rgb[selected_color]

    # Convert the hair color to LAB
    hair_color_lab = cv2.cvtColor(np.uint8([[hair_color_rgb]]), cv2.COLOR_RGB2Lab)[0][0]

    # Replace A and B channels with the new hair color's A and B values
    new_hair_lab = np.zeros_like(image_lab)
    new_hair_lab[:, :, 0] = L_channel  # Keep the original L channel
    new_hair_lab[:, :, 1] = hair_color_lab[1]  # A channel from the new hair color
    new_hair_lab[:, :, 2] = hair_color_lab[2]  # B channel from the new hair color

    # Convert back to RGB color space
    new_hair_rgb = cv2.cvtColor(new_hair_lab, cv2.COLOR_Lab2RGB)

    # Convert mask to 3 channels
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Apply the mask to combine the new hair color with the original image
    final_image = np.where(mask_3d == 1, new_hair_rgb, image_rgb)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the result
    output_path = os.path.join(output_dir, f"{selected_color}_hair.jpg")
    plt.imsave(output_path, final_image)

    print(f"Saved: {output_path}")

if __name__ == '__main__':
    args = parse_arguments()
    apply_hair_color(args)
