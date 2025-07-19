import cv2
import numpy as np
import keyboard

from capturing import VirtualCamera
from overlays import (
    initialize_hist_figure,
    plot_overlay_to_image,
    plot_strings_to_image,
    update_histogram
)
from basics import histogram_figure_numba


def compute_statistics(img):
    # This function calculates basic statistics for each color channel (Red, Green, Blue) of the image
    
    stats = {}  # Dictionary to store stats for each channel
    
    for i, color in enumerate(('R', 'G', 'B')):
        channel = img[:, :, i]  # Extract the i-th color channel from the image
        
        # Find the mode (the most frequent pixel value) in the channel
        values, counts = np.unique(channel, return_counts=True)
        mode_val = values[np.argmax(counts)]
        
        # Calculate entropy, which measures how much "information" or randomness is in the channel
        # First, get the histogram (count of pixel values from 0 to 255)
        hist = np.histogram(channel, bins=256, range=(0, 256))[0]
        prob = hist / np.sum(hist)  # Convert counts to probabilities
        prob = prob[prob > 0]       # Remove zero probabilities to avoid math errors in log calculation
        entropy_val = -np.sum(prob * np.log2(prob))  # Apply the entropy formula
        
        # Save all stats for this color channel
        stats[color] = {
            'Mean': np.mean(channel),        # Average brightness value in this channel
            'Mode': int(mode_val),           # Most common pixel value
            'Std': np.std(channel),          # How spread out the pixel values are (standard deviation)
            'Min': np.min(channel),          # Darkest pixel value
            'Max': np.max(channel),          # Brightest pixel value
            'Entropy': entropy_val           # Amount of randomness or detail in the channel
        }
        
    return stats

def equalize_image(img):
    # This function applies histogram equalization to each color channel of the image
    # Histogram equalization improves the contrast of the image, making dark areas lighter and bright areas clearer

    equalized = np.zeros_like(img)  # Create an empty image array with the same shape and type as input

    for i in range(3):  # Loop over the three color channels: Red, Green, Blue
        # Apply OpenCV's histogram equalization to each channel separately
        equalized[:, :, i] = cv2.equalizeHist(img[:, :, i])

    return equalized  # Return the contrast-enhanced image


def apply_filters(img):
    # Convert the image to grayscale because edge detection works on single channel images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel edge detection:
    # Calculate gradients in x direction (horizontal edges)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    # Calculate gradients in y direction (vertical edges)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Combine horizontal and vertical gradients to get edge strength
    sobel_combined = cv2.magnitude(sobelx, sobely)
    # Normalize and convert to 8-bit image to display properly
    sobel_norm = np.uint8(np.clip(sobel_combined, 0, 255))
    # Convert grayscale edge image back to RGB (3 channels) so it can be displayed with color images
    sobel_colored = cv2.cvtColor(sobel_norm, cv2.COLOR_GRAY2RGB)

    # Apply Gaussian blur to smooth the image and reduce noise
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    # Sharpen the image by emphasizing edges
    kernel = np.array([[0, -1, 0],       # Define sharpening filter kernel
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)  # Apply sharpening filter to the image

    # Return the three processed images: edges, blurred, and sharpened
    return sobel_colored, blurred, sharpened

def linear_transform(img, alpha=1.2, beta=30):
    """Increase contrast and brightness"""
    # Multiply pixel values by alpha (contrast) and add beta (brightness)
    # np.clip ensures pixel values stay within valid range [0,255]
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

def custom_processing(img_source_generator):
    # Initialize the histogram figure and plots (for showing color histograms)
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    key_press_cooldown = 0  # To prevent repeated rapid key press actions

    # Loop over each frame (image) from the video source generator
    for sequence in img_source_generator:
        # Decrease cooldown if active
        if key_press_cooldown > 0:
            key_press_cooldown -= 1

        # If 's' key pressed and cooldown allows, print image statistics
        if keyboard.is_pressed('s') and key_press_cooldown == 0:
            print("Statistics for current frame:")
            stats = compute_statistics(sequence)  # Calculate stats for current frame
            # Print stats for each color channel
            for ch, values in stats.items():
                print(f"{ch} Channel: {values}")
            key_press_cooldown = 5  # Set cooldown to avoid multiple prints quickly

        # Apply contrast and brightness increase
        transformed = linear_transform(sequence)

        # Apply histogram equalization to improve image contrast
        equalized = equalize_image(transformed)

        # Apply filters: Sobel edges, blur, and sharpen
        sobel_img, blurred, sharpened = apply_filters(equalized)

        # Calculate histogram bars for the equalized image's color channels
        r_bars, g_bars, b_bars = histogram_figure_numba(equalized)

        # Update histogram plots with the new histogram data
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot, r_bars, g_bars, b_bars)

        # Overlay the histogram plot onto the equalized image
        equalized_with_hist = plot_overlay_to_image(equalized, fig)

        # Prepare text to display on video frame
        display_text_arr = ["Press 's' for stats", "Showing Equalized + Sobel"]
        # Overlay text onto the Sobel filtered image
        overlay_text = plot_strings_to_image(sobel_img, display_text_arr)

        # Show the processed video frame in a window
        cv2.imshow("Processed Video", overlay_text)

        # Check if 'q' is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

        # Yield the processed frame to be sent to virtual camera or other uses
        yield overlay_text

    # Clean up windows after loop ends
    cv2.destroyAllWindows()

def main():

    # Set resolution and frames per second for the virtual camera
    width = 1280
    height = 720
    fps = 30

    # Create a virtual camera object (simulates a webcam device)
    vc = VirtualCamera(fps, width, height)

    # Try to capture from your real webcam (device index 0)
    try:
        img_source = vc.capture_cv_video(0, bgr_to_rgb=True)
        
        # Start processing frames and send to virtual camera output
        vc.virtual_cam_interaction(
            custom_processing(img_source)
        )

    except KeyboardInterrupt:
        print("Interrupted by user.")  # control c

    finally:
        print("Exiting Virtual Camera Stream.")

if __name__ == "__main__":
    main()  