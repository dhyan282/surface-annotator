frontend/netlify/functions/api.js
const axios = require('axios');

exports.handler = async (event, context) => {
  try {
    const { httpMethod, path, queryStringParameters } = event;
    const body = event.body ? JSON.parse(event.body) : {};
    
    if (httpMethod === 'POST' && path === '/annotate') {
      const response = await axios.post(
        'http://localhost:8000/annotate',
        body
      );
      
      return {
        statusCode: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
        body: JSON.stringify(response.data),
      };
    }
    
    if (httpMethod === 'GET' && path === '/health') {
      return {
        statusCode: 200,
        headers: {
          'Access-Control-Allow-Origin': '*',
        },
        body: JSON.stringify({ status: 'healthy' }),
      };
    }
    
    return {
      statusCode: 404,
      body: JSON.stringify({ error: 'Not found' }),
    };
  } catch (error) {
    console.error('Function error:', error);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: error.message }),
    };
  }
};
