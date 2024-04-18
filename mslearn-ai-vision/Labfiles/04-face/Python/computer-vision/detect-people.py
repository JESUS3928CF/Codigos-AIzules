from dotenv import load_dotenv  # Importa la función load_dotenv desde la biblioteca dotenv
import os  # Importa el módulo os para interactuar con el sistema operativo
from array import array  # Importa el tipo de datos array desde la biblioteca array
from PIL import Image, ImageDraw  # Importa las clases Image y ImageDraw del módulo PIL (Python Imaging Library)
import sys  # Importa el módulo sys para interactuar con el intérprete de Python
import time  # Importa el módulo time para manejar el tiempo
from matplotlib import pyplot as plt  # Importa el módulo pyplot de la biblioteca matplotlib para trazar gráficos
import numpy as np  # Importa el módulo numpy para realizar cálculos numéricos

# Importa el espacio de nombres necesario para interactuar con Azure AI Vision
import azure.ai.vision as sdk  

# Define la función principal del programa
def main():
    global cv_client  # Declara una variable global cv_client

    try:
        # Carga la configuración desde un archivo .env
        load_dotenv()
        ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')  # Obtiene el punto de conexión del servicio de AI desde las variables de entorno
        ai_key = os.getenv('AI_SERVICE_KEY')  # Obtiene la clave del servicio de AI desde las variables de entorno

        # Obtiene la ruta de la imagen a analizar
        image_file = 'images/people.jpg'  # Define una ruta de imagen predeterminada
        if len(sys.argv) > 1:  # Si se proporciona una ruta de imagen como argumento en la línea de comandos, la usa en lugar de la predeterminada
            image_file = sys.argv[1]

        # Autentica el cliente de Azure AI Vision
        cv_client = sdk.VisionServiceOptions(ai_endpoint, ai_key)
        
        # Analiza la imagen
        AnalyzeImage(image_file, cv_client)

    except Exception as ex:  # Captura cualquier excepción que ocurra y la imprime
        print(ex)


# Define la función para analizar la imagen
def AnalyzeImage(image_file, cv_client):
    print('\nAnalyzing', image_file)

    # Especifica las características a recuperar (en este caso, PEOPLE)
    analysis_options = sdk.ImageAnalysisOptions()
        
    features = analysis_options.features = (
        sdk.ImageAnalysisFeature.PEOPLE  # Establece la característica a analizar como PEOPLE (personas)
    )    


    # Obtiene el análisis de la imagen
    image = sdk.VisionSource(image_file)  # Crea un objeto de origen de imagen desde el archivo de imagen
    image_analyzer = sdk.ImageAnalyzer(cv_client, image, analysis_options)  # Crea un analizador de imagen
    result = image_analyzer.analyze()  # Realiza el análisis de la imagen
    
    if result.reason == sdk.ImageAnalysisResultReason.ANALYZED:  # Si la imagen se analizó correctamente
        # Obtiene las personas en la imagen
        if result.people is not None:
            print("\nPeople in image:")  # Imprime un encabezado
            
            # Prepara la imagen para dibujar
            image = Image.open(image_file)  # Abre la imagen usando PIL
            fig = plt.figure(figsize=(image.width/100, image.height/100))  # Crea una figura de matplotlib
            plt.axis('off')  # Desactiva los ejes en la trama
            draw = ImageDraw.Draw(image)  # Crea un objeto de dibujo en la imagen
            color = 'cyan'  # Define el color para dibujar los cuadros delimitadores
            
            # Itera sobre cada persona detectada
            for detected_people in result.people:
                # Dibuja el cuadro delimitador de la persona si la confianza es mayor al 50%
                if detected_people.confidence > 0.5:
                    # Dibuja el cuadro delimitador de la persona
                    r = detected_people.bounding_box
                    bounding_box = ((r.x, r.y), (r.x + r.w, r.y + r.h))  # Define las coordenadas del cuadro delimitador
                    draw.rectangle(bounding_box, outline=color, width=3)  # Dibuja el cuadro delimitador en la imagen
                    
                    # Imprime la confianza de la persona detectada
                    print(" {} (confidence: {:.2f}%)".format(detected_people.bounding_box, detected_people.confidence * 100))
                        
            # Guarda la imagen con las personas detectadas
            plt.imshow(image)  # Muestra la imagen con las personas detectadas
            plt.tight_layout(pad=0)  # Ajusta el diseño de la trama
            outputfile = 'detected_people.jpg'  # Nombre del archivo de salida
            fig.savefig(outputfile)  # Guarda la imagen con las personas detectadas
            print('  Results saved in', outputfile)  # Imprime un mensaje indicando dónde se guardaron los resultados
        
    else:  # Si el análisis falla
        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)  # Obtiene los detalles del error
        print(" Analysis failed.")  # Imprime un mensaje de error
        print("   Error reason: {}".format(error_details.reason))  # Imprime la razón del error
        print("   Error code: {}".format(error_details.error_code))  # Imprime el código de error
        print("   Error message: {}".format(error_details.message))  # Imprime el mensaje de error


# Llama a la función main si este script se ejecuta directamente
if __name__ == "__main__":
    main()
