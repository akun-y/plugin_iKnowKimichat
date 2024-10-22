
import os

from paddleocr_image_recognition import analyze_image_paddle


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__),"images","a.jpg")  
    print(image_path)  
    result = analyze_image_paddle(image_path)
    print(result)

