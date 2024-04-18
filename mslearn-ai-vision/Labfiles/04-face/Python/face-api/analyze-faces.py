# Importar la función load_dotenv para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
# Importar el módulo os para acceder a variables de entorno y operar con archivos
import os
# Importar las clases Image y ImageDraw de la biblioteca PIL para trabajar con imágenes
from PIL import Image, ImageDraw
# Importar la función pyplot de matplotlib para visualizar imágenes
from matplotlib import pyplot as plt

# Importar los espacios de nombres necesarios de Azure Cognitive Services para el reconocimiento facial
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

# Definir la función principal del programa
def main():
    global face_client  # Declarar face_client como una variable global

    try:
        # Obtener la configuración desde el archivo .env
        load_dotenv()
        cog_endpoint = os.getenv('AI_SERVICE_ENDPOINT')  # Obtener el punto final del servicio desde las variables de entorno
        cog_key = os.getenv('AI_SERVICE_KEY')  # Obtener la clave de la API desde las variables de entorno

        # Autenticar el cliente Face utilizando las credenciales
        credentials = CognitiveServicesCredentials(cog_key)
        face_client = FaceClient(cog_endpoint, credentials)

        # Mostrar un menú para seleccionar una función
        print('1: Detect faces\nAny other key to quit')
        command = input('Enter a number:')  # Solicitar al usuario que ingrese un número
        if command == '1':  # Si el usuario ingresa '1', llamar a la función DetectFaces
            DetectFaces(os.path.join('images','l3xoor.jpg'))  # Pasar la ruta de la imagen como argumento

    except Exception as ex:  # Capturar y manejar cualquier excepción
        print(ex)  # Imprimir el mensaje de error si ocurre alguna excepción

# Definir la función para detectar caras en una imagen
def DetectFaces(image_file):
    print('Detecting faces in', image_file)  # Imprimir un mensaje indicando que se están detectando caras en la imagen

    # Especificar las características faciales que se van a recuperar durante la detección
    features = [FaceAttributeType.occlusion,
                FaceAttributeType.blur,
                FaceAttributeType.glasses]

    # Obtener las caras en la imagen utilizando el cliente Face
    with open(image_file, mode="rb") as image_data:
        detected_faces = face_client.face.detect_with_stream(image=image_data,
                                                                return_face_attributes=features,
                                                                return_face_id=False)

        if len(detected_faces) > 0:  # Si se detectan una o más caras en la imagen
            print(len(detected_faces), 'faces detected.')  # Imprimir el número de caras detectadas

            # Preparar la imagen para dibujar los cuadros alrededor de las caras
            fig = plt.figure(figsize=(8, 6))  # Crear una figura con un tamaño específico
            plt.axis('off')  # Desactivar los ejes de la figura
            image = Image.open(image_file)  # Abrir la imagen utilizando PIL
            draw = ImageDraw.Draw(image)  # Crear un objeto ImageDraw para dibujar sobre la imagen
            color = 'lightgreen'  # Definir el color para los cuadros de las caras
            face_count = 0  # Inicializar un contador para el número de caras

            # Dibujar y anotar cada cara detectada en la imagen
            for face in detected_faces:
                # Obtener las propiedades de la cara detectada
                face_count += 1  # Incrementar el contador de caras
                print('\nFace number {}'.format(face_count))  # Imprimir el número de cara

                detected_attributes = face.face_attributes.as_dict()  # Convertir los atributos de la cara a un diccionario
                if 'blur' in detected_attributes:  # Si se detecta borrosidad en la cara
                    print(' - Blur:')  # Imprimir un mensaje indicando la detección de borrosidad
                    for blur_name in detected_attributes['blur']:  # Iterar sobre los diferentes tipos de borrosidad
                        print('   - {}: {}'.format(blur_name, detected_attributes['blur'][blur_name]))  # Imprimir los detalles de la borrosidad

                if 'occlusion' in detected_attributes:  # Si se detecta ocultamiento en la cara
                    print(' - Occlusion:')  # Imprimir un mensaje indicando la detección de ocultamiento
                    for occlusion_name in detected_attributes['occlusion']:  # Iterar sobre los diferentes tipos de ocultamiento
                        print('   - {}: {}'.format(occlusion_name, detected_attributes['occlusion'][occlusion_name]))  # Imprimir los detalles del ocultamiento

                if 'glasses' in detected_attributes:  # Si se detectan gafas en la cara
                    print(' - Glasses:{}'.format(detected_attributes['glasses']))  # Imprimir un mensaje indicando la detección de gafas

                # Dibujar un cuadro alrededor de la cara detectada en la imagen
                r = face.face_rectangle  # Obtener el rectángulo delimitador de la cara
                bounding_box = ((r.left, r.top), (r.left + r.width, r.top + r.height))  # Definir las coordenadas del cuadro delimitador
                draw.rectangle(bounding_box, outline=color, width=5)  # Dibujar el cuadro delimitador alrededor de la cara
                annotation = 'Face number {}'.format(face_count)  # Crear una etiqueta para anotar la cara
                plt.annotate(annotation,(r.left, r.top), backgroundcolor=color)  # Anotar la cara en la imagen con la etiqueta

            # Guardar la imagen con los cuadros alrededor de las caras detectadas
            plt.imshow(image)  # Mostrar la imagen con los cuadros de las caras
            outputfile = 'detected_faces.jpg'  # Definir el nombre del archivo de salida
            fig.savefig(outputfile)  # Guardar la figura como una imagen

            print('\nResults saved in', outputfile)  # Imprimir un mensaje indicando que los resultados se han guardado en el archivo

# Comprobar si este script es el script principal
if __name__ == "__main__":
    main()  # Llamar a la función principal si este script es el script principal