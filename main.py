import cv2
import numpy as np
import os
import pytesseract
from collections import OrderedDict

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)

raw_data_set_folder = "raw_data_set"
os.makedirs(raw_data_set_folder, exist_ok=True)

prep_data_set_folder = "prep_data_set"
os.makedirs(prep_data_set_folder, exist_ok=True)

cont_data_set_folder = "cont_data_set"
os.makedirs(cont_data_set_folder, exist_ok=True)

got_num_folder = "got_num"
os.makedirs(got_num_folder, exist_ok=True)

no_num_folder = "no_num"
os.makedirs(no_num_folder, exist_ok=True)

roi_folder = os.path.join(output_folder, "roi")
os.makedirs(roi_folder, exist_ok=True)



binary_roi_folder = "binary_roi"
os.makedirs(binary_roi_folder, exist_ok=True)

otsu_roi_folder = "otsu_roi"
os.makedirs(otsu_roi_folder, exist_ok=True)

dilated_roi_folder = "dilated_roi"
os.makedirs(dilated_roi_folder, exist_ok=True)

opened_roi_folder = "no_num"
os.makedirs(opened_roi_folder, exist_ok=True)

closed_roi_folder = "no_num"
os.makedirs(closed_roi_folder, exist_ok=True)

def remove_distortion(image, k1=-0.45, k2=0.01, p1=0.001, p2=0.001):
    h, w = image.shape[:2]

    camera_matrix = np.array([[w, 0, w / 2],
                             [0, h, h / 2],
                             [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float32)

    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

    return undistorted_image

def save_to_folder(image, folder, name):
    path = os.path.join(folder, name)
    cv2.imwrite(path, image)
    print(f"Изображение сохранено: {path}")

# 6, 7, 8, 11
def process_roi(roi, config='--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789'): 
    text = pytesseract.image_to_string(roi, config=config).strip()
    return text if len(text) >= 5 and text.isdigit() else None

def process_and_save_roi(roi, folder, prefix, index, text, save_roi_steps=False):
    if text:
        if save_roi_steps:
            save_to_folder(roi, folder, f"{prefix}_{index}_roi.png")
        return True
    return False

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    save_to_folder(gray, output_folder, "1_gray.png")

    blurred = cv2.GaussianBlur(gray, (9, 9), 7)
    save_to_folder(blurred, output_folder, "2_blurred.png")

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 5) # 31.7
    save_to_folder(thresh, output_folder, "3_thresh.png")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    save_to_folder(opened, output_folder, "4_opened.png")

    dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(opened, dilated_kernel, iterations=4)
    save_to_folder(dilated, output_folder, "5_dilated.png")

    return dilated

def find_and_draw_digits(raw_image, processed_image, image_counter, save_results=False):
    all_image = raw_image.copy()
    output_image = raw_image.copy()

    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(all_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if save_results:
        save_to_folder(all_image, cont_data_set_folder, f"{image_counter}.png")

    detected_numbers = []

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 1000:
            continue

        # Текущая зона
        roi = raw_image[y:y + h, x:x + w]

        # Серое изображение
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Бинаризация через Otsu и Clahe
        roi_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi_gray) # 2.0 ; 8, 8
        _, roi_otsu = cv2.threshold(roi_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        text_otsu = process_roi(roi_otsu)
        save_to_folder(roi_otsu, otsu_roi_folder, f"1_otsu_{i}_{image_counter}.png")
        if process_and_save_roi(roi_otsu, otsu_roi_folder, "1_otsu", i, text_otsu, save_roi_steps=False):
            print("Номер найден на OTSU")
            detected_numbers.append(text_otsu)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Морфология для Otsu
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph_otsu = cv2.morphologyEx(roi_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph_otsu = cv2.morphologyEx(morph_otsu, cv2.MORPH_OPEN, kernel, iterations=1)

        text_otsu_morph = process_roi(morph_otsu)
        if process_and_save_roi(morph_otsu, roi_folder, "2_otsu_morph", i, text_otsu_morph, save_roi_steps=True):
            print("Номер найден на OTSU + MORPH")
            detected_numbers.append(text_otsu_morph)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        morph_otsu = cv2.morphologyEx(morph_otsu, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        text_otsu_morph = process_roi(morph_otsu)
        if process_and_save_roi(morph_otsu, roi_folder, "2_otsu_morph", i, text_otsu_morph, save_roi_steps=True):
            print("Номер найден на OTSU + MORPH")
            detected_numbers.append(text_otsu_morph)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Совмещение Otsu и Adaptive
        roi_adaptive = cv2.adaptiveThreshold(roi_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 31, 5)
        combined_roi = cv2.bitwise_and(roi_otsu, roi_adaptive)

        text_combined = process_roi(combined_roi)
        if process_and_save_roi(combined_roi, roi_folder, "3_otsu", i, text_combined, save_roi_steps=False):
            print("Номер найден на COMBINED")
            detected_numbers.append(text_combined)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Морфология после Combined
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_roi = cv2.morphologyEx(combined_roi, cv2.MORPH_CLOSE, kernel, iterations=1)
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, kernel, iterations=1)

        text_commor = process_roi(morph_roi)
        if process_and_save_roi(morph_roi, roi_folder, "4_otsu_improved", i, text_commor, save_roi_steps=True):
            print("Номер найден на MORPH COMBINED")
            detected_numbers.append(text_commor)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Чистый Adaptive
        binary_roi = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7) # 21.7
        text_binary = process_roi(binary_roi)
        save_to_folder(binary_roi, binary_roi_folder, f"5_binary_{i}_{image_counter}.png")
        if process_and_save_roi(binary_roi, binary_roi_folder, "5_binary", i, text_binary, save_roi_steps=False):
            print("Номер найден на BINARY")
            detected_numbers.append(text_binary)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Дилатация после Adaptive
        dilated_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) #3.3
        dilated_roi = cv2.dilate(binary_roi, dilated_kernel, iterations=2) #3
        text_dilate = process_roi(dilated_roi)
        #save_to_folder(dilated_roi, dilated_roi_folder, f"6_dilated_{i}_{image_counter}.png")
        if process_and_save_roi(dilated_roi, roi_folder, "6_dilated", i, text_dilate, save_roi_steps=False):
            print("Номер найден на DILATED")
            detected_numbers.append(text_dilate)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Morph Close после Adaptive
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        text_close = process_roi(morph_roi)
        #save_to_folder(morph_roi, close_roi_folder, f"7_morph_close_{i}_{image_counter}.png")
        if process_and_save_roi(morph_roi, roi_folder, "7_morph_close", i, text_close, save_roi_steps=False):
            print("Номер найден на CLOSE")
            detected_numbers.append(text_close)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

        # Morph Open после Morph Close
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph_roi = cv2.morphologyEx(morph_roi, cv2.MORPH_OPEN, open_kernel, iterations=1)
        text_open = process_roi(morph_roi)
        #save_to_folder(morph_roi, open_roi_folder, f"8_morph_open_{i}_{image_counter}.png")
        if process_and_save_roi(morph_roi, roi_folder, "8_morph_open", i, text_open, save_roi_steps=False):
            print("Номер найден на OPEN")
            detected_numbers.append(text_open)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            continue

    # Финальная проверка
    if detected_numbers:
        save_to_folder(output_image, got_num_folder, f"{image_counter}.png")
        print(f"Найдены номера в изображении {image_counter}: {detected_numbers}")
    else:
        save_to_folder(output_image, no_num_folder, f"{image_counter}.png")
        print(f"Номера не найдены в изображении {image_counter}.")

    return list(OrderedDict.fromkeys(detected_numbers))

def process_video(video_path, save_raw=False, save_prep=False, prep=False):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка загрузки видео. Проверьте путь и попробуйте снова.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    frame_count = 0
    paused = False

    save_count_1 = 1
    save_count_2 = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или произошла ошибка чтения.")
            break

        undistorted_frame = remove_distortion(frame)
        
        cv2.imshow("Undistorted Video", undistorted_frame)
        
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break

        elif key == ord(' '):
            if save_raw:
                print("Сохранение оригинальных кадров включено")
                save_to_folder(undistorted_frame, raw_data_set_folder, f"{save_count_1}.png")
                save_count_1 += 1
            else:
                print("Сохранение оригинальных кадров отключено")
            
            if prep:
                print("Предобработка кадров включена")
                processed_image = preprocess_image(undistorted_frame)
                if save_prep:
                    print("Сохранение предобработанных кадров включена")
                    save_to_folder(processed_image, prep_data_set_folder, f"{save_count_2}.png")
                    save_count_2 += 1
                else:
                    print("Сохранение предобработанных кадров выключено")
            else:
                print("Предобработка кадров отключена")

        elif key == ord('p'):
            paused = not paused
            if paused:
                print("Видео на паузе. Нажмите 'p', чтобы продолжить.")
            else:
                print("Воспроизведение возобновлено.")
                
    cap.release()
    cv2.destroyAllWindows()

def process_single_image(image_path, save_results=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка загрузки изображения: {image_path}")
        return

    undistorted_image = remove_distortion(image)

    processed_image = preprocess_image(undistorted_image)

    detected_numbers = find_and_draw_digits(undistorted_image, processed_image, "single_image", save_results)

    if detected_numbers:
        print(f"Найдены номера: {detected_numbers}")
    else:
        print("Номера не найдены.")

def process_prep_images():
    steps = False
    raw_images = sorted(os.listdir(raw_data_set_folder), key=lambda x: int(x.split('.')[0]))
    prep_images = sorted(os.listdir(prep_data_set_folder), key=lambda x: int(x.split('.')[0]))
    
    if len(raw_images) != len(prep_images):
        print("Ошибка: количество файлов в raw_data_set и prep_data_set не совпадает.")
        return
    
    image_counter = 1

    for raw_name, prep_name in zip(raw_images, prep_images):
        raw_path = os.path.join(raw_data_set_folder, raw_name)
        prep_path = os.path.join(prep_data_set_folder, prep_name)
        
        raw_image = cv2.imread(raw_path)
        prep_image = cv2.imread(prep_path, cv2.IMREAD_GRAYSCALE)
        
        if raw_image is None or prep_image is None:
            print(f"Ошибка загрузки изображений: {raw_name} или {prep_name}")
            continue
        
        print(f"Обработка изображения {image_counter}: {raw_name}")
        detected_numbers = find_and_draw_digits(raw_image, prep_image, image_counter, False)
        
        if steps:
            user_input = input("Нажмите Enter для продолжения или 'q' для выхода: ")
            if user_input.lower() == 'q':
                break
        
        image_counter += 1

def process_selected_images():
    raw_image_name = input("Введите имя сырого изображения (например, 1.png): ")
    prep_image_name = input("Введите имя предобработанного изображения (например, 1.png): ")

    raw_path = os.path.join(raw_data_set_folder, raw_image_name)
    prep_path = os.path.join(prep_data_set_folder, prep_image_name)

    raw_image = cv2.imread(raw_path)
    prep_image = cv2.imread(prep_path, cv2.IMREAD_GRAYSCALE)

    if raw_image is None or prep_image is None:
        print(f"Ошибка загрузки изображений: {raw_image_name} или {prep_image_name}")
        return

    print(f"Обработка выбранных изображений: {raw_image_name} и {prep_image_name}")
    detected_numbers = find_and_draw_digits(raw_image, prep_image, "selected", False)

def main():
    while True:
        print("Выберите режим:")
        print("1. Кадр: первичная предобработка")
        print("2. Кадр: поиск и выделение номера")
        print("3. Кадр: обработка выбранного")
        print("4. Обработка отдельного изображения")
        print("5. Выход")
        choice = input("Введите номер режима: ")
        
        if choice == '1':
            video_path = input("Введите путь к видео: ")
            process_video(video_path, save_raw=True, save_prep=True, prep=True)
        elif choice == '2':
            process_prep_images()
        elif choice == '3':
            process_selected_images()
        elif choice == '4':
            image_path = input("Введите путь к изображению: ")
            process_single_image(image_path, save_results=True)
        elif choice == '5':
            print("Выход из программы.")
            break
        else:
            print("Неверный выбор. Попробуйте снова.")
            
if __name__ == "__main__":
    main()