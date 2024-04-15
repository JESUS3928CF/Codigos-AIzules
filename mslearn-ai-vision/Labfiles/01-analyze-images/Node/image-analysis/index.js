const { ImageAnalysisClient } = require('@azure-rest/ai-vision-image-analysis');
const createClient = require('@azure-rest/ai-vision-image-analysis').default;
const { AzureKeyCredential } = require('@azure/core-auth');

require('dotenv').config(); // Carga las variables de entorno desde el archivo .env 

const endpoint = process.env['VISION_ENDPOINT']; // Define la URL del punto de enlace para el servicio de análisis de imágenes de Azure
const key = process.env['VISION_KEY']; // Define la clave de acceso para autenticación
const credential = new AzureKeyCredential(key);

const client = new createClient(endpoint, credential);

const imageUrl = 'https://aka.ms/azsdk/image-analysis/sample.jpg';
const features = [
    'Caption',
    'DenseCaptions',
    'Objects',
    'People',
    'Read',
    'SmartCrops',
    'Tags',
];

async function analyzeImageFromUrl() {
    const result = await client.path('/imageanalysis:analyze').post({
        body: {
            url: imageUrl,
        },
        queryParameters: {
            features: features,
            'smartCrops-aspect-ratios': [0.9, 1.33],
        },
        contentType: 'application/json',
    });

    console.log('Image analysis result:', result.body);
}

analyzeImageFromUrl();
