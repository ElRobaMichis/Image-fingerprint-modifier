from image_processing import ImageProcessor
from ui import ImageProcessorUI

def main():
    # Create the image processor
    processor = ImageProcessor()
    
    # Create and run the UI
    app = ImageProcessorUI(processor)
    app.run()

if __name__ == "__main__":
    main()