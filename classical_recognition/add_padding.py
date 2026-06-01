import cv2
import os
import glob

def add_padding_to_folder(input_dir, output_dir, padding=20):
    """
    Dodaje padding do wszystkich obrazów w folderze.
    
    :param input_dir: Ścieżka do folderu z oryginalnymi wzorcami.
    :param output_dir: Ścieżka do folderu, gdzie zostaną zapisane nowe obrazy.
    :param padding: Rozmiar marginesu w pikselach (z każdej strony).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    search_pattern = os.path.join(input_dir, '*.[pP][nN][gG]')
    image_paths = glob.glob(search_pattern)

    if not image_paths:
        print(f"Nie znaleziono obrazów w folderze: {input_dir}")
        return

    for img_path in image_paths:
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Nie udało się wczytać pliku: {img_path}")
            continue

        background_color = [0, 0, 0] 

        padded_img = cv2.copyMakeBorder(
            img, 
            padding, padding, padding, padding, 
            cv2.BORDER_CONSTANT, 
            value=background_color
        )

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        cv2.imwrite(output_path, padded_img)
        print(f"Zapisano obraz z paddingiem: {output_path}")

folder_wejsciowy = "classical_recognition/patterns"
folder_wyjsciowy = "classical_recognition/patterns_padded"

add_padding_to_folder(folder_wejsciowy, folder_wyjsciowy, padding=15)