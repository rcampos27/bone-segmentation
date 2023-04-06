import os
from collections import Counter
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import numpy as np
from pathlib import Path
from datetime import date

input_path = 'D:\\TCC\\data\\input'
output_path = 'D:\\TCC\\data\\output'
img_size = 512

#! Colors (B, G, R)
red = (0, 0, 255)
orange = (0, 134, 255)
yellow = (0, 215, 255)
green = (0, 255, 81)
blue = (255, 161, 0)
black = (0, 0, 0)
white = (255, 255, 255)


def saveAsPNG(path, image_number):
    dicom = pydicom.dcmread(path)

    # * Implementation Version Names: imagem de exame -> 'CV_102' (219908 arquivos); laudo -> 'CYBERMEDDCM30' (209 arquivos)
    ivn = dicom.file_meta.get((0x0002, 0x0013), 'CYBERMEDDCM30').value
    if ivn == 'CYBERMEDDCM30':
        return

    # gender = (dicom.PatientSex)
    # genders.append(gender)

    # dt = dicom.PatientBirthDate
    # if (dt):
    #     y = int(dt[0:4])
    #     m = int(dt[4:6])
    #     d = int(dt[6:8])
    #     today = date.today()
    #     age = today.year - y - ((today.month, today.day) < (m, d))
    #     if (age < 18): return
    #     else:
    #         ages.append(age)

    img = dicom.pixel_array.astype(float)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    img = img.astype(np.uint8)

    # ? Transforma a intensidade dos pixels da imagem (grayscale) em seus valores correspondentes na escala Hounsfield (HU)
    hu = apply_modality_lut(dicom.pixel_array, dicom)

    mask = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)
    limiar = np.zeros((img.shape[1], img.shape[0], 3), np.uint8)
    for row in range(0, len(hu)):
        for col in range(0, len(hu[row])):
            value = hu[row][col]
            if value < 148:
                # ? BACKGROUND
                limiar[row][col] = black
            elif value >= 148 and value <= 667:
                # ? OSSO ESPONJOSO (III|IV / SOFT)
                limiar[row][col] = blue
            elif value >= 668 and value <= 1000:
                # ? OSSO COMPACTO (II|III / NORMAL)
                limiar[row][col] = green
                if value > 850:
                    mask[row][col] = white
            elif value >= 1001 and value <= 1770:
                # ? OSSO COMPACTO (I / HARD)
                limiar[row][col] = yellow
                mask[row][col] = white
            elif value >= 1771 and value <= 2850:
                # ? ESMALTE
                limiar[row][col] = orange
                mask[row][col] = white
            else:
                limiar[row][col] = red
                mask[row][col] = white

    kernel7 = np.ones((7, 7),np.uint8)
    kernel9 = np.ones((9, 9),np.uint8)

    canny = cv2.Canny(mask, 10, 200)
    canny = cv2.GaussianBlur(canny, (7, 7), 0)
    canny = cv2.dilate(canny, kernel9, iterations=3)
    canny = cv2.erode(canny, kernel7, iterations=3)
    morph = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel7)

    contours, _  = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = cv2.drawContours(morph, contours, -1, white, cv2.FILLED)

    contour = cv2.cvtColor(contour, cv2.COLOR_GRAY2BGR)
    filtered_limiar = cv2.bitwise_and(limiar, contour)

    h = round((img_size - img.shape[0])/2)
    w = round((img_size - img.shape[1])/2)
    # img_src = cv2.copyMakeBorder(src=img, top=h, bottom=h, left=w, right=w, borderType=cv2.BORDER_CONSTANT)
    # cv2.imwrite(f'{img_name}.png', img_src)


    #! Save Hounsfield values array, padded till 512x512, this file will later be loaded and converted into a PyTorch tensor and serve as input for the neural networks
    hu = cv2.copyMakeBorder(src=hu, top=h, bottom=h, left=w,
                            right=w, borderType=cv2.BORDER_CONSTANT, value=hu.min())
    input_name = f'{output_path}\\input\\{image_number}'
    np.save(f'{input_name}.npy', hu)

    # Save expected segmentation
    img_limiar = cv2.copyMakeBorder(
        src=filtered_limiar, top=h, bottom=h, left=w, right=w, borderType=cv2.BORDER_CONSTANT, value=black)
    exp_name = f'{output_path}\\expected\\{image_number}'
    cv2.imwrite(f'{exp_name}.png', img_limiar)

    # Save the hard tissue mask generated 
    img_contour = cv2.copyMakeBorder(
        src=contour, top=h, bottom=h, left=w, right=w, borderType=cv2.BORDER_CONSTANT, value=black)
    msk_name = f'{output_path}\\mask\\{image_number}'
    cv2.imwrite(f'{msk_name}.png', img_contour)

    dicom.clear()


if __name__ == "__main__":
    patient = 0
    dcm_sizes = []
    total_processed = 0
    stop = False
    counter = 0
    Path(output_path).mkdir(parents=True, exist_ok=True)

    patient = 0
    for root, dirs, files in os.walk(input_path):
        id = int(patient/3)
        for name in files:
            counter += 1
            path = os.path.join(root, name)
            print(f'{path}\\{id}\\{counter}')
            saveAsPNG(f'{path}', counter)
        dcm_sizes.append(counter)
        patient += 1
        if stop:
            break

result = list(Counter(dcm_sizes).items())
print(sorted(result, key=lambda x: x[1]))
print(f'{patient} images')
