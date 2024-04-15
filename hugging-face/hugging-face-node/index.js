const axios = require('axios');
const request = require('request-promise');
const fs = require('fs');

const API_KEY = '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';
const IMAGE_URL = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'; // Replace with your image URL
const OUTPUT_FILE = 'caption.txt';

function getImageData(url) {
  return new Promise((resolve, reject) => {
    request({ url, encoding: null })
      .then(buffer => resolve(buffer))
      .catch(error => reject(error));
  });
}

function getCaption(imageData) {

  console.log(imageData)
  return axios.post('https://api.clarifai.com/v2/models/general-en', {
    inputs: [{ data: { image: { source: imageData } } }]
  }, {
    headers: {
      Authorization: `Bearer ${API_KEY}`
    }
  });
}

(async () => {
  try {
    const imageData = await getImageData(IMAGE_URL);
    const response = await getCaption(imageData);
    const concepts = response.data.outputs[0].data.concepts;
    const caption = concepts.map(concept => concept.name).join(', '); // Join concepts with commas
    fs.writeFileSync(OUTPUT_FILE, caption);
    console.log(`Caption saved to ${OUTPUT_FILE}`);
  } catch (error) {
    console.error("error");
  }
})();
