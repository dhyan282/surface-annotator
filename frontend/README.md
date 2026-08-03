frontend/README.md
# Surface Auto-Annotator Frontend

This is the React frontend for the Surface Auto-Annotator application, deployed on Netlify.

## Features

- Modern React interface with Tailwind CSS
- Real-time WebSocket communication
- Drag-and-drop image upload
- Progress tracking for annotation tasks
- Responsive design

## Tech Stack

- React 18+
- Tailwind CSS
- WebSocket for real-time updates
- Axios for API calls
- Netlify Functions for serverless backend

## Deployment

Built and deployed using Netlify CI/CD pipeline with serverless functions.

## Running Locally

```bash
cd frontend
npm install
npm start
```

## Netlify Configuration

- The frontend is statically built
- API calls are handled by Netlify Functions
- WebSocket connections are established client-side
