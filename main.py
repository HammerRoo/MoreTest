import cv2
import numpy as np
import os
import pytesseract
from collections import OrderedDict

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
roi_folder = os.path.join(output_folder, "roi")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(roi_folder, exist_ok=True)

def save_image(image, name):
    path = os.path.join(output_folder, name)
    cv2.imwrite(path, image)
    print(f"Изображение сохранено: {path}")

def save_roi(image, name):
    path = os.path.join(roi_folder, name)
    cv2.imwrite(path, image)
    print(f"ROI сохранен: {path}")

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_image(gray, "1_gray.png")

    blurred = cv2.GaussianBlur(gray, (11, 11), 7) # 6
    save_image(blurred, "2_blurred.png")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    save_image(thresh, "3_thresh.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_image(opened, "4_opened.png")


    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) # 7, 7
    dilated = cv2.dilate(opened, dilated_kernel, iterations=5) # 4
    save_image(dilated, "5_dilated.png")

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    save_image(output_image, "6_result.jpg")

    return dilated

def find_and_draw_digits(image, processed_image):
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = image.copy()
    detected_numbers = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 100:
            continue

        roi = image[y:y+h, x:x+w]
        text_clean = pytesseract.image_to_string(
            roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_clean) >= 7 and text_clean.isdigit():
            detected_numbers.append(text_clean)
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print(f"Номер найден на original_roi: {text_clean}")
            continue
        save_roi(roi, f"{i}_original_roi.png")

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        text_gray = pytesseract.image_to_string(
            roi_gray, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_gray) >= 7 and text_gray.isdigit():
            detected_numbers.append(text_gray)
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print(f"Номер найден на gray_roi: {text_gray}")
            continue
        save_roi(roi_gray, f"{i}_gray_roi.png")

        binary_roi = cv2.adaptiveThreshold(
            roi_gray, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            21,
            10
        )
        text_binary = pytesseract.image_to_string(
            binary_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()
        if len(text_binary) >= 7 and text_binary.isdigit():
            detected_numbers.append(text_binary)
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print(f"Номер найден на binary_roi: {text_binary}")
            continue
        save_roi(binary_roi, f"{i}_binary_roi.png")

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel, iterations=2)
        save_roi(morph_roi, f"{i}_morph_close_roi.png")
        
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, kernel, iterations=1)
        save_roi(morph_roi, f"{i}_morph_open_roi.png")

        scale_factor = max(2, int(300 / max(w, h)))
        resized_roi = cv2.resize(morph_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        save_roi(resized_roi, f"{i}_resized_roi.png")

        text_processed = pytesseract.image_to_string(
            resized_roi, 
            config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
        ).strip()

        if len(text_processed) >= 6 and text_processed.isdigit():
            detected_numbers.append(text_processed)
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"Номер найден после предобработки: {text_processed}")
        else:
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 1)

    return list(OrderedDict.fromkeys(detected_numbers))

def process_video():
    video_path = input("Введите путь к видео: ")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    frame_skip = 10
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break
        
        frame_count += 1
        
        cv2.imshow("Video", frame)
        
        if frame_count % frame_skip == 0:
            processed_image = preprocess_image(frame)
            detected_numbers = find_and_draw_digits(frame, processed_image)
            
            if detected_numbers:
                print(f"Кадр {frame_count}: Номер вагона виден: {detected_numbers}")
            else:
                print(f"Кадр {frame_count}: Номер вагона не виден")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_selected_frame():
    video_path = input("Введите путь к видео: ")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    frame_count = 0
    paused = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break
        
        frame_count += 1
        
        cv2.imshow("Video", frame)
        
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break

        elif key == ord(' '):
            processed_image = preprocess_image(frame)
            # detected_numbers = find_and_draw_digits(frame, processed_image)
            
            # if detected_numbers:
            #     print(f"Кадр {frame_count}: Номер вагона виден: {detected_numbers}")
            # else:
            #     print(f"Кадр {frame_count}: Номер вагона не виден")

        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Видео на паузе. Нажмите 'p', чтобы продолжить.")
            else:
                print("Воспроизведение возобновлено.")
                
    cap.release()
    cv2.destroyAllWindows()

def process_image():
    while True:
        image_path = input("Введите путь к изображению (или 'exit' для выхода): ")
        
        if image_path.lower() == 'exit':
            print("Выход из программы.")
            break
        
        image = cv2.imread(image_path)
        
        if image is None:
            print("Ошибка загрузки изображения. Проверьте путь и попробуйте снова.")
            continue
        
        processed_image = preprocess_image(image)
        
        detected_numbers = find_and_draw_digits(image, processed_image)

        if detected_numbers:
            print("Номер вагона виден: ", detected_numbers)
        else:
            print("Номер вагона не виден")

def main():
    while True:
        print("Выберите режим:")
        print("1. Обработка изображения")
        print("2. Обработка видео")
        print("3. Запуск видео и обработка выбранных кадров.")
        print("4. Выход")
        choice = input("Введите номер режима: ")
        
        if choice == '1':
            process_image()
        elif choice == '2':
            process_video()
        elif choice == '3':
            process_selected_frame()
        elif choice == '4':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            
if __name__ == "__main__":
    main()