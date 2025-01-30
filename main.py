#Bibliotecas 
import numpy as np
import cv2
import time
from csv import DictWriter

##carrega arquivo
video = cv2.VideoCapture('Videos/13020022_3840_2160_30fps.mp4')

#cria variaveis de altura e largura
h, w = None, None

#carega arquivos do modelo
with open('yoloDados/YoloNames.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('YoloDados/yolov3.cfg',
                                     'YoloDados/yolov3.weights')

#camadas de saída 
layers_names_all = network.getLayerNames()
layers_names_output = [layers_names_all[int(i)-1] 
                       for i in network.getUnconnectedOutLayers()]

#Detectação
probability_minimum = 0.5
threshold = 0.3

#cores para caixa de objetos 
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

while True: 
    #captura de frames 
    ret, frame = video.read()

    if not ret:
        break
    if w is None or h is None:
        h,w = frame.shape[:2]

    #pré-processamento da imagem
    blob = cv2.dnn.blobFromImage(frame, 1/ 255.0, (416, 416), 
                                 swapRB=True, crop=False)
    # Executa o modelo
    network.setInput(blob)
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()
  # Listas para armazenar os objetos detectados
    bounding_boxes = []
    confidences = []
    class_numbers = []

        # Processa as saídas do modelo
    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

        # Aplica a supressão de não máxima (NMS) para filtrar detecções ruins
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box_current = colours[class_numbers[i]].tolist()
                
                # Desenha a caixa no frame
            cv2.rectangle(frame, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            colour_box_current, 2)
                # Texto com o nome do objeto e acurácia
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                    confidences[i])

                # Coloca o texto sobre o objeto detectado
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)


    #Exibe vidfeo com objetos 
    cv2.namedWindow('YOLO V3 - VIDEO', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO V3 - VIDEO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
#libera recursos
video.release()
cv2.destroyAllWindows()