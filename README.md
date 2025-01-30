YOLO Object Detection com OpenCV

Este projeto implementa a detecção de objetos em um vídeo usando YOLO (You Only Look Once) e OpenCV.

Requisitos

Antes de rodar o projeto, certifique-se de ter os seguintes pacotes instalados:

pip install opencv-python numpy

Também é necessário baixar os arquivos de configuração do YOLO:

yolov3.cfg

yolov3.weights

YoloNames.names

Coloque esses arquivos na pasta yoloDados/.

Como Usar

Substitua o caminho do arquivo de vídeo na linha:

video = cv2.VideoCapture('caminho/do/video.mp4')

Execute o script:

python main.py

Para encerrar a execução, pressione q.

Funcionalidades

Detecta objetos em um vídeo utilizando YOLOv3.

Exibe as caixas delimitadoras e a confiança da detecção.

Estrutura do Projeto
/
|-- main.py  # Script principal
|-- yoloDados/
|   |-- yolov3.cfg
|   |-- yolov3.weights
|   |-- YoloNames.names
|-- Videos/
|   |-- arquivo.mp4


Licença

Este projeto é de uso livre. Sinta-se à vontade para modificar e melhorar! 🚀
