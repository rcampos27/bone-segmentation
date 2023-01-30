import os
from collections import Counter
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import numpy as np
from pathlib import Path
from datetime import date

input_paths = ['E:\\TCC\\_data\\input\\ClinicaUnioeste', 'E:\\TCC\\_data\\input\\ClinicaUltraface']

output_path = 'E:\\TCC\\_data\\output'

ivns = []
ages = []
genders = []
#! Colors
purple = (140, 71, 84)
orange = (33,36,231)
red = (76, 158, 242)
white = (255, 255, 255)

def saveAsPNG(path, file_number, image_number):
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
                mask[row][col] = (0,0,0)            
            elif value >= 148 and value <= 667 :
                #? OSSO ESPONJOSO (III|IV / SOFT)
                mask[row][col] = purple
            elif value >= 668 and value <= 1000 :
                #? OSSO COMPACTO (II|III / NORMAL)
                # Maize Crayola
                mask[row][col] = (168, 137, 4)
            elif value >= 1001 and value <= 1770 :
                #? OSSO COMPACTO (I / HARD)
                # Maize Crayola
                mask[row][col] = orange
            elif value >= 1771 and value <= 2850 :
                #? ESMALTE 
                mask[row][col] = red
            else:
                # White
                mask[row][col] = white

    # ? Sobrepõe a imagem original e a máscara da segmentação
    img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    final = cv2.addWeighted(img2, 0.8, mask, 0.6, 0.0)

    img_name = f'{output_path}\\{file_number}-{image_number}'

    # cv2.imwrite(f'{img_name}.png', img)
    # cv2.imwrite(f'{img_name}_final.png', final)
    cv2.imwrite(f'{img_name}_mask.png', mask)

    dicom.clear()


if __name__ == "__main__":
    patient = 0
    dcm_sizes = []
    total_processed = 0
    stop = False
    Path(output_path).mkdir(parents=True, exist_ok=True)

    for input_path in input_paths:

        patient = 0
        for root, dirs, files in os.walk(input_path):
            counter = 0
            id = int(patient/3)
            for name in files:
                counter += 1
                path = os.path.join(root, name)
                print(f'{path}\\{id}\\{counter}')
                saveAsPNG(f'{path}', id, counter)
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
