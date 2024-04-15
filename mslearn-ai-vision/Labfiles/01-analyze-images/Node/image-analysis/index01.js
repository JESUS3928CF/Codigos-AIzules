// Importación de módulos necesarios
const createClient = require('@azure-rest/ai-vision-image-analysis').default; // Importa el módulo para crear el cliente para análisis de imágenes
const { AzureKeyCredential } = require('@azure/core-auth'); // Importa el módulo para manejar las credenciales de Azure

// Load the .env file if it exists
require('dotenv').config(); // Carga las variables de entorno desde el archivo .env si existe

// Definición de variables de configuración
const endpoint =
    process.env['VISION_ENDPOINT']; // Define la URL del punto de enlace para el servicio de análisis de imágenes de Azure
const key = process.env['VISION_KEY']; // Define la clave de acceso para autenticación
const credential = new AzureKeyCredential(key); // Crea una instancia de AzureKeyCredential utilizando la clave de acceso

// Creación del cliente para interactuar con el servicio de análisis de imágenes de Azure
const client = createClient(endpoint, credential);

// Definición de las características que se desean analizar en la imagen
const features = [
    'Caption',
    'DenseCaptions',
    'Objects',
    'People',
    'Read',
    'SmartCrops',
    'Tags',
];

// URL de la imagen que se va a analizar
const imageUrl = 'https://aka.ms/azsdk/image-analysis/sample.jpg';

// Función asincrónica para analizar la imagen desde una URL
async function analyzeImageFromUrl() {
    // Envío de la solicitud de análisis de imagen al servicio de Azure
    const result = await client.path('imageanalysis:analyze').post({
        body: {
            url: imageUrl, // Especifica la URL de la imagen a analizar en el cuerpo de la solicitud
        },
        queryParameters: {
            features: features, // Especifica las características que se desean analizar en la imagen
            'smartCrops-aspect-ratios': [0.9, 1.33], // Parámetro opcional para análisis de recortes inteligentes con aspectos específicos
        },
        contentType: 'application/json', // Especifica el tipo de contenido de la solicitud como JSON
    });

    // Procesamiento de la respuesta del servicio
    const iaResult = result.body;

    // Impresión de los resultados del análisis de imagen
    console.log(`Model Version: ${iaResult.modelVersion}`); // Imprime la versión del modelo utilizado para el análisis
    console.log(`Image Metadata: ${JSON.stringify(iaResult.metadata)}`); // Imprime los metadatos de la imagen
    if (iaResult.captionResult) {
        console.log(
            `Caption: ${iaResult.captionResult.text} (confidence: ${iaResult.captionResult.confidence})`
        ); // Imprime la descripción de la imagen y la confianza asociada
    }
    if (iaResult.denseCaptionsResult) {
        // Si se detectaron descripciones densas en la imagen, imprime cada una de ellas
        iaResult.denseCaptionsResult.values.forEach((denseCaption) =>
            console.log(`Dense Caption: ${JSON.stringify(denseCaption)}`)
        );
    }
    if (iaResult.objectsResult) {
        // Si se detectaron objetos en la imagen, imprime cada uno de ellos
        iaResult.objectsResult.values.forEach((object) =>
            console.log(`Object: ${JSON.stringify(object)}`)
        );
    }
    if (iaResult.peopleResult) {
        // Si se detectaron personas en la imagen, imprime cada una de ellas
        iaResult.peopleResult.values.forEach((person) =>
            console.log(`Person: ${JSON.stringify(person)}`)
        );
    }
    if (iaResult.readResult) {
        // Si se detectó texto en la imagen, imprime cada bloque de texto
        iaResult.readResult.blocks.forEach((block) =>
            console.log(`Text Block: ${JSON.stringify(block)}`)
        );
    }
    if (iaResult.smartCropsResult) {
        // Si se detectaron recortes inteligentes en la imagen, imprime cada uno de ellos
        iaResult.smartCropsResult.values.forEach((smartCrop) =>
            console.log(`Smart Crop: ${JSON.stringify(smartCrop)}`)
        );
    }
    if (iaResult.tagsResult) {
        // Si se detectaron etiquetas en la imagen, imprime cada una de ellas
        iaResult.tagsResult.values.forEach((tag) =>
            console.log(`Tag: ${JSON.stringify(tag)}`)
        );
    }
}

// Llamada a la función de análisis de imagen
analyzeImageFromUrl();
