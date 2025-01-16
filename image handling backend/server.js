const express = require('express');
const multer = require('multer');
const axios = require('axios');
const sharp = require('sharp');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for image upload
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowedTypes.includes(file.mimetype)) {
      return cb(new Error('Invalid file type. Only JPG, JPEG, and PNG are allowed.'));
    }
    cb(null, true);
  },
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB limit
  }
});

// Configuration for Python API
const PYTHON_API_URL = 'http://localhost:5000/predict';

// Utility function to verify image visibility
async function verifyImage(buffer) {
  try {
    const metadata = await sharp(buffer).metadata();
    
    // Check if image dimensions are valid
    if (metadata.width < 50 || metadata.height < 50) {
      throw new Error('Image dimensions too small. Minimum 50x50 pixels required.');
    }

    // Analyze image statistics to check if it's not completely black/white/transparent
    const stats = await sharp(buffer).stats();
    
    // Check if image is not completely black or white
    const isBlank = stats.channels.every(channel => {
      const mean = channel.mean;
      return mean < 5 || mean > 250;
    });

    if (isBlank) {
      throw new Error('Image appears to be blank or not visible.');
    }

    return true;
  } catch (error) {
    throw new Error(`Image validation failed: ${error.message}`);
  }
}

// Route to handle image upload and processing
app.post('/api/detect-pest', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image file provided' });
    }

    // Verify image visibility
    await verifyImage(req.file.buffer);

    // Create form data for Python API
    const formData = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype });
    formData.append('file', blob, req.file.originalname);

    // Send to Python API
    const response = await axios.post(PYTHON_API_URL, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });

    res.json(response.data);
  } catch (error) {
    if (error.response) {
      // Error from Python API
      res.status(error.response.status).json({
        error: error.response.data.error || 'Error from pest detection service'
      });
    } else {
      // Local error (validation, etc.)
      res.status(400).json({
        error: error.message || 'Error processing image'
      });
    }
  }
});

// Error handling middleware
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        error: 'File size too large. Maximum size is 5MB.'
      });
    }
  }
  res.status(500).json({
    error: error.message || 'Internal server error'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Node.js server running on port ${PORT}`);
});