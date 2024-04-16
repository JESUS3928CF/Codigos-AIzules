const fs = require('fs');
const axios = require('axios');

require('dotenv').config();

// Rest of your code

// Ruta de la imagen que deseas enviar al modelo
const imagePath = './test-images/IMG_TEST_2.jpg';

// Claves y puntos finales del modelo
const predictionKey = process.env.PredictionKey;
const predictionEndpoint = process.env.PredictionEndpoint;
const projectId = process.env.ProjectID;
const publishedModelName = process.env.PublishedModelName;

// Lee la imagen como un búfer
const imageBuffer = fs.readFileSync(imagePath);

/// Realiza una solicitud HTTP POST al modelo con el archivo local
axios
    .post(
        `${predictionEndpoint}customvision/v3.0/Prediction/${projectId}/classify/iterations/${publishedModelName}/image`,
        imageBuffer,
        {
            headers: {
                'Prediction-key': predictionKey,
                'Content-Type': 'application/octet-stream',
            },
        }
    )
    .then((response) => {
        const predictions = response.data.predictions;
        console.log('Predicciones imagen local:');
        predictions.forEach((prediction) => {
            console.log(
                `${prediction.tagName}: ${Math.round(
                    prediction.probability * 100
                )}%`
            );
        });
    })
    .catch((error) => {
        console.error(
            'Error al realizar la predicción del archivo local:',
            error.message
        );
    });


/// Realiza una solicitud HTTP POST al modelo
axios
    .post(
        `${predictionEndpoint}customvision/v3.0/Prediction/${projectId}/classify/iterations/${publishedModelName}/url`,
        {
            Url: 'https://concepto.de/wp-content/uploads/2023/01/avion.jpg',
        },
        {
            headers: {
                'Prediction-key': predictionKey,
                'Content-Type': 'application/json',
            },
        }
    )
    .then((response) => {
        const predictions = response.data.predictions;
        console.log('Predicciones imagen url:');
        predictions.forEach((prediction) => {
            console.log(
                `${prediction.tagName}: ${Math.round(
                    prediction.probability * 100
                )}%`
            );
        });
    })
    .catch((error) => {
        console.error('Error al realizar la predicción:', error.message);
    });


