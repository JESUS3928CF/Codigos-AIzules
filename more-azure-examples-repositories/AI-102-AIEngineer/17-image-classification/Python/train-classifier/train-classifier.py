from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient  # Importar clase CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region  # Importar modelos necesarios
from msrest.authentication import ApiKeyCredentials  # Importar clase ApiKeyCredentials para autenticación
import time  # Importar módulo time para retardos
import os  # Importar módulo os para operaciones con el sistema de archivos

def main():  # Definir la función principal como punto de entrada
    global training_client  # Declarar variable global para el cliente de entrenamiento
    global custom_vision_project  # Declarar variable global para el proyecto de Custom Vision

    try:
        # Cargar variables de entorno desde el archivo .env
        from dotenv import load_dotenv
        load_dotenv()

        # Obtener configuraciones de variables de entorno
        training_endpoint = os.getenv('TrainingEndpoint')
        training_key = os.getenv('TrainingKey')
        project_id = os.getenv('ProjectID')

        # Autenticar un cliente para la API de entrenamiento usando credenciales de clave API
        credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        training_client = CustomVisionTrainingClient(training_endpoint, credentials)

        # Obtener el proyecto de Custom Vision usando el ID del proyecto
        custom_vision_project = training_client.get_project(project_id)

        # Subir y etiquetar imágenes usando la función Upload_Images
        Upload_Images('more-training-images')

        # Entrenar el modelo usando la función Train_Model
        Train_Model()

    except Exception as ex:  # Manejar excepciones (errores)
        print(ex)

def Upload_Images(folder):  # Definir la función Upload_Images para subir imágenes
    print("Subiendo imágenes...")

    # Obtener etiquetas asociadas con el proyecto de Custom Vision
    tags = training_client.get_tags(custom_vision_project.id)

    # Iterar sobre cada etiqueta
    for tag in tags:
        print(f"Etiqueta: {tag.name}")  # Imprimir el nombre de la etiqueta actual

        # Iterar sobre cada archivo de imagen en la subcarpeta de la etiqueta
        for image in os.listdir(os.path.join(folder, tag.name)):
            image_data = open(os.path.join(folder, tag.name, image), "rb").read()  # Leer datos de la imagen como binarios

            # Crear un objeto ImageFileCreateEntry y subir la imagen con el ID de la etiqueta
            training_client.create_images_from_data(custom_vision_project.id, image_data, [tag.id])

def Train_Model():  # Definir la función Train_Model para entrenar el modelo
    print("Entrenando...")

    # Iniciar el proceso de entrenamiento para el proyecto de Custom Vision
    iteration = training_client.train_project(custom_vision_project.id)

    # Monitorear el estado del entrenamiento hasta que se complete
    while (iteration.status != "Completed"):
        iteration = training_client.get_iteration(custom_vision_project.id, iteration.id)
        print(f"Estado del entrenamiento: {iteration.status} ...")
        time.sleep(5)  # Introducir un retraso entre comprobaciones de estado

    print("¡Modelo entrenado!")


if __name__ == "__main__":  # Verificar si el script se ejecuta directamente
    main()  # Llamar a la función principal para iniciar la ejecución