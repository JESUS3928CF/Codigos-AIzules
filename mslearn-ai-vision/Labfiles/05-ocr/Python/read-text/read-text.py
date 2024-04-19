from dotenv import load_dotenv  # Importa la función `load_dotenv` de la librería `dotenv`
import os  # Importa el módulo `os` para acceder a variables de entorno y otras funcionalidades del sistema operativo
import time  # Importa el módulo `time` para trabajar con el tiempo
from PIL import Image, ImageDraw  # Importa clases de la librería `PIL` para trabajar con imágenes
from matplotlib import pyplot as plt  # Importa la función `pyplot` de la librería `matplotlib` para visualización de datos

# Importa clases y funciones específicas de la librería Azure AI Vision
from azure.ai.vision.imageanalysis import ImageAnalysisClient  
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def main():  # Define la función principal `main`

    global cv_client  # Define una variable global `cv_client`

    try:  # Manejo de excepciones para capturar errores
        # Obtener configuración desde archivos de entorno
        load_dotenv()  # Carga las variables de entorno desde un archivo `.env`
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')  # Obtiene la URL del servicio de AI desde las variables de entorno
        ai_key = os.getenv('AI_SERVICE_KEY')  # Obtiene la clave de acceso al servicio de AI desde las variables de entorno

        # Autentica el cliente de Azure AI Vision
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Menú para funciones de lectura de texto
        print('\n1: Use Read API for image (Lincoln.jpg)\n2: Read handwriting (Note.jpg)\nAny other key to quit\n')
        command = input('Enter a number:')  # Solicita al usuario ingresar un número
        if command == '1':  # Si el usuario ingresa '1'
            image_file = os.path.join('images','Lincoln.jpg')  # Define la ruta de la imagen a procesar
            GetTextRead(image_file)  # Llama a la función `GetTextRead` para procesar la imagen
        elif command =='2':  # Si el usuario ingresa '2'
            image_file = os.path.join('images','Note.jpg')  # Define la ruta de la imagen a procesar
            GetTextRead(image_file)  # Llama a la función `GetTextRead` para procesar la imagen
                

    except Exception as ex:  # Captura cualquier excepción y la almacena en `ex`
        print(ex)  # Imprime la excepción

def GetTextRead(image_file):  # Define la función `GetTextRead` que toma la ruta de un archivo de imagen como argumento
    print('\n')

    # Abre el archivo de imagen
    with open(image_file, "rb") as f:
            image_data = f.read()

    # Utiliza la función `analyze` para leer el texto en la imagen
    result = cv_client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]
    )

    # Muestra la imagen y sobrepone el texto extraído
    if result.read is not None:  # Si se detectó texto en la imagen
        print("\nText:")  # Imprime un encabezado

        # Abre la imagen
        image = Image.open(image_file)
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        plt.axis('off')
        draw = ImageDraw.Draw(image)
        color = 'cyan'

        # Recorre las líneas de texto detectadas en la imagen
        for line in result.read.blocks[0].lines:
            print(f"  {line.text}")  # Imprime el texto de cada línea

            drawLinePolygon = True

            r = line.bounding_polygon
            bounding_polygon = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))

            print("   Bounding Polygon: {}".format(bounding_polygon))

            for word in line.words:
                r = word.bounding_polygon
                bounding_polygon = ((r[0].x, r[0].y),(r[1].x, r[1].y),(r[2].x, r[2].y),(r[3].x, r[3].y))
                print(f"    Word: '{word.text}', Bounding Polygon: {bounding_polygon}, Confidence: {word.confidence:.4f}")

                drawLinePolygon = False
                draw.polygon(bounding_polygon, outline=color, width=3)

            if drawLinePolygon:
                draw.polygon(bounding_polygon, outline=color, width=3)

        print("\n")

        for line in result.read.blocks[0].lines:
            print(f"  {line.text}") 

        

        # Guarda la imagen
        plt.imshow(image)
        plt.tight_layout(pad=0)
        outputfile = 'text.jpg'
        fig.savefig(outputfile)
        print('\n  Results saved in', outputfile)

if __name__ == "__main__":  # Verifica si este script es el punto de entrada principal
    main()  # Llama a la función principal `main` para comenzar la ejecución del programa
