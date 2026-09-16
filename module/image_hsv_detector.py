import cv2
import numpy as np
import sys
import os

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV manually for better understanding"""
    r, g, b = r/255.0, g/255.0, b/255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    
    # Hue calculation
    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    elif cmax == b:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    # Saturation
    s = 0 if cmax == 0 else (diff / cmax) * 100
    
    # Value
    v = cmax * 100
    
    return int(h), int(s), int(v)

def main():
    # Check for image path argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to your image: ").strip().strip('"\'')
    
    # Validate image path
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at '{image_path}'")
        print("Please provide a valid image path (jpg, png, etc.)")
        sys.exit(1)
    
    # Load the image
    print("Loading image...")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Error: Could not load the image. Please check the file format.")
        sys.exit(1)
    
    print(f"✅ Image loaded successfully! Resolution: {image.shape[1]}x{image.shape[0]}")
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("• Left Click on any pixel to get HSV + RGB values")
    print("• Move mouse while holding left button for live preview")
    print("• Press 'q' or ESC to quit")
    print("="*60 + "\n")
    
    # Create a window
    cv2.namedWindow("Color Picker - Click on image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Color Picker - Click on image", 900, 700)
    
    # Variables for color swatch and info
    selected_color = None
    selected_hsv = None
    selected_rgb = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_color, selected_hsv, selected_rgb
        
        if x < 0 or y < 0 or x >= image.shape[1] or y >= image.shape[0]:
            return
        
        # Get BGR color from image (OpenCV uses BGR by default)
        b, g, r = image[y, x]
        
        # Convert to HSV using OpenCV
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = hsv_pixel
        
        # Also compute manually for verification
        h_manual, s_manual, v_manual = rgb_to_hsv(r, g, b)
        
        selected_rgb = (r, g, b)
        selected_hsv = (h, s, v)
        selected_color = (b, g, r)  # for drawing
        
        if event == cv2.EVENT_LBUTTONDOWN or flags & cv2.EVENT_FLAG_LBUTTON:
            print("\n" + "-"*50)
            print(f"📍 Pixel Position: ({x}, {y})")
            print(f"RGB: R={r:3d}, G={g:3d}, B={b:3d}")
            print(f"HSV (OpenCV): H={h:3d}, S={s:3d}, V={v:3d}")
            print(f"HSV (Manual): H={h_manual:3d}, S={s_manual:3d}, V={v_manual:3d}")
            print(f"Hex Color: #{r:02X}{g:02X}{b:02X}")
            print("-"*50)
    
    # Set mouse callback
    cv2.setMouseCallback("Color Picker - Click on image", mouse_callback)
    
    print("Window opened. Click anywhere on the image to pick colors!\n")
    
    while True:
        display_img = image.copy()
        
        # Draw color swatch and info if a color is selected
        if selected_color is not None:
            # Draw a large color swatch
            swatch_size = 120
            swatch = np.full((swatch_size, swatch_size, 3), selected_color, dtype=np.uint8)
            
            # Put swatch on top-left corner
            display_img[20:20+swatch_size, 20:20+swatch_size] = swatch
            
            # Draw border around swatch
            cv2.rectangle(display_img, (18, 18), (20+swatch_size+2, 20+swatch_size+2), (255, 255, 255), 2)
            
            # Display text information
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_img, f"RGB: {selected_rgb}", (20, 170), font, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, f"HSV: {selected_hsv}", (20, 200), font, 0.6, (255, 255, 255), 2)
            cv2.putText(display_img, "Click to pick new color", (20, 240), font, 0.5, (200, 200, 200), 1)
        
        # Show the image
        cv2.imshow("Color Picker - Click on image", display_img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
    
    cv2.destroyAllWindows()
    print("\n👋 Color picker closed. Thank you!")

if __name__ == "__main__":
    # Optional: Allow running with image as command line argument
    # Example: python color_picker_hsv.py "path/to/image.jpg"
    main()