from dotenv import load_dotenv  # Importa la función load_dotenv desde la librería dotenv
import os  # Importa el módulo os para interactuar con el sistema operativo
from PIL import Image, ImageDraw  # Importa las clases Image y ImageDraw de la librería Pillow (PIL)
import sys  # Importa el módulo sys para interactuar con el intérprete de Python
from matplotlib import pyplot as plt  # Importa la interfaz pyplot de Matplotlib
from azure.core.exceptions import HttpResponseError  # Importa la excepción HttpResponseError de Azure Core
import requests  # Importa la librería requests para hacer solicitudes HTTP

# Import namespaces
# Importa las clases y constantes necesarias del paquete de Azure para el análisis de imágenes
from azure.ai.vision.imageanalysis import ImageAnalysisClient  # Clase para interactuar con el servicio de análisis de imágenes de Azure
from azure.ai.vision.imageanalysis.models import VisualFeatures  # Constantes que representan características visuales para solicitar al analizar una imagen
from azure.core.credentials import AzureKeyCredential  # Clase para autenticar las solicitudes a los servicios de Azure mediante una clave de autenticación


# La función main() es la entrada principal del script
def main():
    global cv_client  # Define cv_client como una variable global

    try:
        load_dotenv()  # Carga las variables de entorno desde el archivo .env
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')  # Obtiene el endpoint del servicio de AI desde las variables de entorno
        ai_key = os.getenv('AI_SERVICE_KEY')  # Obtiene la clave del servicio de AI desde las variables de entorno

        # Define el nombre de archivo de la imagen a analizar (puede ser proporcionado como argumento en la línea de comandos)
        image_file = 'images/m93zvm.jpg'
        if len(sys.argv) > 1:
            image_file = sys.argv[1]

        with open(image_file, "rb") as f:
            image_data = f.read()  # Lee el contenido binario del archivo de imagen

        # Autentica el cliente de Azure AI Vision
        cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key)
        )

        # Analiza la imagen
        AnalyzeImage(image_file, image_data, cv_client)

        # Elimina el fondo de la imagen utilizando la API de Azure
        BackgroundForeground(ai_endpoint, ai_key, image_file)

        # Elimina el fondo de la imagen local utilizando la API de Azure
        BackgroundForegroundLocal(ai_endpoint, ai_key, image_data)

    except Exception as ex:  # Maneja cualquier excepción
        print(ex)  # Imprime la excepción si ocurre alguna

#/ Este código realiza el análisis de una imagen utilizando un cliente de Azure AI Vision, y luego muestra los resultados del análisis, incluyendo subtítulos, etiquetas, objetos y personas detectadas en la imagen. También guarda una versión de la imagen con los objetos y personas detectadas resaltadas.
def AnalyzeImage(image_filename, image_data, cv_client):
    print('\nAnalyzing image...')

    try:
        # Realiza el análisis de la imagen con las características especificadas
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE],
        )

    except HttpResponseError as e:
        # Maneja errores de HTTP
        print(f"Status code: {e.status_code}")
        print(f"Reason: {e.reason}")
        print(f"Message: {e.error.message}")

    # Muestra los resultados del análisis
    # Obtiene subtítulos de la imagen
    if result.caption is not None:
        print("\nCaption:")
        print(" Caption: '{}' (confidence: {:.2f}%)".format(result.caption.text, result.caption.confidence * 100))

    # Obtiene subtítulos densos de la imagen
    if result.dense_captions is not None:
        print("\nDense Captions:")
        for caption in result.dense_captions.list:
            print(" Caption: '{}' (confidence: {:.2f}%)".format(caption.text, caption.confidence * 100))

    # Obtiene etiquetas de la imagen
    if result.tags is not None:
        print("\nTags:")
        for tag in result.tags.list:
            print(" Tag: '{}' (confidence: {:.2f}%)".format(tag.name, tag.confidence * 100))

    #/ Obtiene objetos en la imagen
    if result.objects is not None:
        print("\nObjects in image:")

        #/ Prepara la imagen para dibujar
        # Abre la imagen utilizando Pillow (PIL) y la asigna a la variable 'image'
        image = Image.open(image_filename)
        # Crea una figura de Matplotlib con el tamaño calculado en función del ancho y alto de la imagen
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        # Desactiva los ejes de la figura de Matplotlib para que no se muestren
        plt.axis('on')
        # Crea un objeto de dibujo de imagen usando la imagen abierta, lo que permite dibujar en la imagen
        draw = ImageDraw.Draw(image)
        # Define el color que se utilizará para dibujar los cuadros delimitadores y las etiquetas, en este caso, 'cyan' (cian)
        color = 'red'

        for detected_object in result.objects.list:
            # Imprime el nombre del objeto
            print(" {} (confidence: {:.2f}%)".format(detected_object.tags[0].name, detected_object.tags[0].confidence * 100))
            
            #/ Dibuja el cuadro delimitador del objeto
            # Obtiene las coordenadas del cuadro delimitador del objeto detectado
            r = detected_object.bounding_box
            # Crea una tupla que representa las coordenadas del cuadro delimitador en forma de un rectángulo
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height)) 
            # Dibuja un rectángulo utilizando las coordenadas del cuadro delimitador
            draw.rectangle(bounding_box, outline=color, width=3)
            # Agrega una etiqueta al cuadro delimitador del objeto
            plt.annotate(detected_object.tags[0].name, (r.x, r.y), backgroundcolor=color)

        #/ Guarda la imagen anotada
        # Muestra la imagen con los cuadros delimitadores y etiquetas agregados
        plt.imshow(image)
        # Ajusta el diseño de la figura para que los elementos estén más cerca
        plt.tight_layout(pad=0)
        # Especifica el nombre del archivo de salida donde se guardará la imagen anotada
        outputfile = 'objects.jpg'
        # Guarda la figura actual (con la imagen anotada) en un archivo JPEG con el nombre especificado
        fig.savefig(outputfile)
        # Imprime un mensaje indicando que los resultados han sido guardados en el archivo especificado
        print('  Results saved in', outputfile)

    #/ Obtiene personas en la imagen
    # Verifica si hay personas detectadas en la imagen
    if result.people is not None:
        # Imprime un mensaje indicando que se encontraron personas en la imagen
        print("\nPeople in image:")

        # Prepara la imagen para dibujar
        # Abre la imagen utilizando Pillow (PIL)
        image = Image.open(image_filename)
        # Crea una nueva figura de Matplotlib con un tamaño ajustado en función del tamaño de la imagen
        fig = plt.figure(figsize=(image.width/100, image.height/100))
        # Desactiva los ejes de la figura para que no se muestren
        plt.axis('off')
        # Crea un objeto de dibujo de imagen para poder dibujar sobre la imagen
        draw = ImageDraw.Draw(image)
        # Define el color que se utilizará para dibujar los cuadros delimitadores, en este caso, 'cyan' (cian)
        color = 'cyan'

        # Itera sobre cada persona detectada en la imagen
        for detected_people in result.people.list:
            # Obtiene las coordenadas del cuadro delimitador de la persona
            r = detected_people.bounding_box
            # Define las coordenadas del cuadro delimitador como una tupla que representa un rectángulo
            bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
            # Dibuja un rectángulo alrededor de la persona utilizando las coordenadas del cuadro delimitador
            draw.rectangle(bounding_box, outline=color, width=3)

        # Muestra la imagen con los cuadros delimitadores de las personas dibujados
        plt.imshow(image)
        # Ajusta el diseño de la figura para que los elementos estén más cerca
        plt.tight_layout(pad=0)
        # Especifica el nombre del archivo de salida donde se guardará la imagen anotada
        outputfile = 'people.jpg'
        # Guarda la figura actual (con la imagen anotada) en un archivo JPEG con el nombre especificado
        fig.savefig(outputfile)
        # Imprime un mensaje indicando que los resultados han sido guardados en el archivo especificado
        print('  Results saved in', outputfile)


#/ Este código realiza una solicitud a la API de Azure Computer Vision para eliminar el fondo de una imagen o generar un mate frontal. Se utiliza la URL de la imagen como entrada para el procesamiento. Una vez que se obtiene la respuesta, que contiene la imagen procesada, se guarda en un archivo local llamado "backgroundForeground.png". Finalmente, se imprime un mensaje indicando que los resultados han sido guardados en el archivo especificado.
def BackgroundForeground(endpoint, key, image_file):
    # Define la versión de la API y el modo
    api_version = "2023-02-01-preview"
    mode = "foregroundMatting"  # Puede ser "foregroundMatting" o "backgroundRemoval"
    
    #/ Elimina el fondo de la imagen o genera un mate frontal
    print('\nRemoving background from image...')
        
    # Construye la URL para la solicitud de eliminación de fondo
    url = "{}computervision/imageanalysis:segment?api-version={}&mode={}".format(endpoint, api_version, mode)

    # Define los encabezados de la solicitud HTTP
    headers = {
        "Ocp-Apim-Subscription-Key": key,  # Clave de suscripción de Azure para autorización
        "Content-Type": "application/json"  # Tipo de contenido de la solicitud
    }

    #/ En la original pide parámetros aca solo al url 
    # URL de la imagen para la eliminación de fondo
    image_url = "https://fifpro.org/media/5chb3dva/lionel-messi_imago1019567000h.jpg?raw=true"

    # Cuerpo de la solicitud JSON que contiene la URL de la imagen
    body = {
        "url": image_url,
    }
        
    # Realiza una solicitud POST a la API de Azure Computer Vision para eliminar el fondo
    response = requests.post(url, headers=headers, json=body)

    # Obtiene el contenido de la respuesta, que es la imagen procesada
    image = response.content
    # Guarda la imagen procesada en un archivo PNG
    with open("backgroundForeground.png", "wb") as file:
        file.write(image)
    # Imprime un mensaje indicando que los resultados han sido guardados en el archivo especificado
    print('  Results saved in backgroundForeground.png \n')



#! Elimina el fondo de una imagen local usando la API de Azure Cognitive Services Computer Vision.
#/ Este código envía una solicitud a la API de Azure Computer Vision para eliminar el fondo de una imagen local. Utiliza datos de imagen binarios en lugar de una URL de imagen. La imagen procesada se guarda en un archivo llamado "backgroundForegroundLocal.png". Si se produce algún error durante el proceso, se maneja adecuadamente e imprime un mensaje de error correspondiente.
def BackgroundForegroundLocal(endpoint, key, image_data):

    #/ Define la versión de la API y el modo
    api_version = "2023-02-01-preview"
    mode = "backgroundRemoval"  # Puede ser "foregroundMatting" o "backgroundRemoval"

    #/ Eliminar el fondo de la imagen
    print('\nEliminando el fondo de la imagen...')

    # Construye la URL para la solicitud de eliminación de fondo
    url = "{}computervision/imageanalysis:segment?api-version={}&mode={}".format(endpoint, api_version, mode)

    # Define los encabezados de la solicitud HTTP

    # El cambio en el encabezado Content-Type a "application/octet-stream" indica que la solicitud ahora está enviando datos de imagen binarios en lugar de una URL de imagen.
    headers = {
        "Ocp-Apim-Subscription-Key": key,  # Clave de suscripción de Azure para autorización
        "Content-Type": "application/octet-stream"  # Tipo de contenido de la solicitud (datos de imagen binarios)
    }

    try:
        # Realiza una solicitud POST a la API de Azure Computer Vision para eliminar el fondo
        response = requests.post(url, headers=headers, data=image_data)

        # Verifica si la solicitud fue exitosa (código de estado HTTP 200)
        if response.status_code == 200:
            # Si la solicitud fue exitosa, obtiene la imagen procesada de la respuesta
            image = response.content
            # Guarda la imagen procesada en un archivo PNG llamado "backgroundForegroundLocal.png"
            with open("backgroundForegroundLocal.png", "wb") as file:
                file.write(image)
            # Imprime un mensaje indicando que los resultados han sido guardados en el archivo especificado
            print('  Results saved in backgroundForegroundLocal.png \n')
        else:
            # Si la solicitud no fue exitosa, imprime un mensaje de error con el código de estado HTTP
            print(f"Error: {response.status_code}")

    except FileNotFoundError:
        # Si se produce un error de archivo no encontrado, imprime un mensaje de error
        print("Error: No se encontró el archivo de imagen.")

if __name__ == "__main__":
    main()  # Llama a la función 'main' para ejecutar el análisis de imágenes
