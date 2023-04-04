import os
from collections import Counter
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import numpy as np
from pathlib import Path
from datetime import date

input_path = 'E:\\_TCC\\_data\\input\\training'

output_path = 'E:\\_TCC\\_data\\output\\training'

ivns = []
ages = []
genders = []
#! Colors (B, G, R)
red = (0, 0, 255)
orange = (0, 134, 255)
yellow = (0, 215, 255)
green = (0, 255, 81)
blue = (255, 161, 0) 

orange_pastel = (88, 176, 255)
yellow_pastel = (88, 229, 255)
blue_pastel = (242, 186, 92)

black = (0, 0, 0)

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
    for row in range(0, len(hu)):
        for col in range(0, len(hu[row])):
            value = hu [row][col]
            if value < 148:
                #? BACKGROUND
                mask[row][col] = black       
            elif value >= 148 and value <= 667 :
                #? OSSO ESPONJOSO (III|IV / SOFT)
                mask[row][col] = blue
            elif value >= 668 and value <= 1000 :
                #? OSSO COMPACTO (II|III / NORMAL)
                mask[row][col] = green
            elif value >= 1001 and value <= 1770 :
                #? OSSO COMPACTO (I / HARD)
                mask[row][col] = yellow
            elif value >= 1771 and value <= 2850 :
                #? ESMALTE 
                mask[row][col] = orange
            else:
                mask[row][col] = red

    img_name = f'{output_path}\\{image_number}'

    h = round((512 - img.shape[0])/2)
    w = round((512 - img.shape[1])/2)
    img_src = cv2.copyMakeBorder(src=img, top=h, bottom=h, left=w, right=w, borderType=cv2.BORDER_CONSTANT) 
    img_msk = cv2.copyMakeBorder(src=mask, top=h, bottom=h, left=w, right=w, borderType=cv2.BORDER_CONSTANT) 

    cv2.imwrite(f'{img_name}.png', img_src)
    cv2.imwrite(f'{img_name}_e.png', img_msk)

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
            # if patient > 6:
            #     stop = True
            #     break
        dcm_sizes.append(counter)
        patient += 1
        if stop:
            break

result = list(Counter(dcm_sizes).items())
final_ivns = list(Counter(ivns).items())
final_ages = sorted(list(Counter(ages).items()), key=lambda tup:tup[0])
final_genders = list(Counter(genders).items())
print(sorted(result, key=lambda x: x[1]))
print(f'{patient} images')
print('IVNS')
print(final_ivns)
print('Ages')
print(final_ages)
print('Genders')
print(final_genders)
