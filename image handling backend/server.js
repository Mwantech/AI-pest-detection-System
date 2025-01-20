const express = require('express');
const multer = require('multer');
const axios = require('axios');
const sharp = require('sharp');
const path = require('path');
const cors = require('cors');
const FormData = require('form-data');

const app = express();
const PORT = 3000;

// Enhanced CORS configuration
app.use(cors({
  origin: ['http://localhost:8081', 'exp://localhost:8081'], // Add your Expo development URLs
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json());
app.use(express.static('public'));

// Enhanced multer configuration
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
    fileSize: 10 * 1024 * 1024 // 10MB limit
  }
}).single('image');

// Configuration
const config = {
  PYTHON_API_URL: 'http://localhost:5000/api/predict',
  MIN_IMAGE_DIMENSION: 50,
  MAX_IMAGE_DIMENSION: 4096,
  COMPRESSION_QUALITY: 80,
  MAX_RETRIES: 3,
  RETRY_DELAY: 1000, // 1 second
};

// Enhanced image validation function
async function validateAndProcessImage(buffer) {
  try {
    // Get image metadata
    const metadata = await sharp(buffer).metadata();
    
    // Log image details for debugging
    console.log('Image metadata:', {
      width: metadata.width,
      height: metadata.height,
      format: metadata.format,
      size: buffer.length
    });
    
    // Validate dimensions with more specific error messages
    if (metadata.width < config.MIN_IMAGE_DIMENSION || metadata.height < config.MIN_IMAGE_DIMENSION) {
      throw new Error(
        `Image dimensions (${metadata.width}x${metadata.height}) are too small. ` +
        `Minimum ${config.MIN_IMAGE_DIMENSION}x${config.MIN_IMAGE_DIMENSION} pixels required.`
      );
    }
    
    if (metadata.width > config.MAX_IMAGE_DIMENSION || metadata.height > config.MAX_IMAGE_DIMENSION) {
      // Instead of throwing error, let's resize the image
      const processedBuffer = await sharp(buffer)
        .resize(config.MAX_IMAGE_DIMENSION, config.MAX_IMAGE_DIMENSION, {
          fit: 'inside',
          withoutEnlargement: true
        })
        .jpeg({ quality: config.COMPRESSION_QUALITY })
        .toBuffer();
      
      return processedBuffer;
    }

    // Process image with more forgiving parameters
    const processedBuffer = await sharp(buffer)
      .jpeg({ 
        quality: config.COMPRESSION_QUALITY,
        force: false // Don't force JPEG if it's another format
      })
      .toBuffer();

    // Verify processed image size
    if (processedBuffer.length > 10 * 1024 * 1024) {
      // If still too large, compress further
      return await sharp(processedBuffer)
        .jpeg({ quality: 60 }) // Lower quality for large images
        .toBuffer();
    }

    return processedBuffer;
  } catch (error) {
    // Enhanced error logging
    console.error('Image validation error:', {
      originalError: error.message,
      bufferSize: buffer ? buffer.length : 'no buffer',
      timestamp: new Date().toISOString()
    });
    
    throw new Error(`Image processing failed: ${error.message}`);
  }
}


// Utility function for retrying failed requests
async function retryRequest(fn, retries = config.MAX_RETRIES) {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, config.RETRY_DELAY));
    }
  }
}

// Enhanced route handler for pest detection
app.post('/api/detect-pest', (req, res) => {
  upload(req, res, async (err) => {
    try {
      // Handle multer errors
      if (err instanceof multer.MulterError) {
        if (err.code === 'LIMIT_FILE_SIZE') {
          return res.status(400).json({
            error: 'File size too large. Maximum size is 5MB.'
          });
        }
        throw err;
      } else if (err) {
        throw err;
      }

      // Validate request
      if (!req.file) {
        return res.status(400).json({ error: 'No image file provided' });
      }

      // Process and validate image
      const processedImageBuffer = await validateAndProcessImage(req.file.buffer);

      // Prepare form data for Python API
      const formData = new FormData();
      formData.append('file', processedImageBuffer, {
        filename: 'processed_image.jpg',
        contentType: 'image/jpeg'
      });

      // Send to Python API with retry mechanism
      const response = await retryRequest(async () => {
        const result = await axios.post(config.PYTHON_API_URL, formData, {
          headers: {
            ...formData.getHeaders(),
            'Accept': 'application/json'
          },
          timeout: 30000, // 30 second timeout
        });

        // Validate response format
        if (!result.data || (!result.data.predictions && !result.data.error)) {
          throw new Error('Invalid response format from pest detection service');
        }

        return result;
      });

      // Format response
      res.json({
        success: true,
        predictions: response.data.predictions,
        pest_info: response.data.pest_info,
        metadata: {
          processed: true,
          original_size: req.file.size,
          processed_size: processedImageBuffer.length,
          timestamp: new Date().toISOString()
        }
      });

    } catch (error) {
      // Enhanced error handling
      const errorResponse = {
        success: false,
        error: error.message || 'An error occurred during processing',
        code: error.code || 'UNKNOWN_ERROR'
      };

      if (error.response) {
        // Error from Python API
        errorResponse.code = 'PYTHON_API_ERROR';
        errorResponse.error = error.response.data.error || 'Error from pest detection service';
        res.status(error.response.status).json(errorResponse);
      } else if (error.request) {
        // Network error
        errorResponse.code = 'NETWORK_ERROR';
        errorResponse.error = 'Unable to reach pest detection service';
        res.status(503).json(errorResponse);
      } else {
        // Local processing error
        res.status(400).json(errorResponse);
      }

      // Log error for debugging
      console.error('Pest Detection Error:', {
        timestamp: new Date().toISOString(),
        error: error.message,
        stack: error.stack,
        code: errorResponse.code
      });
    }
  });
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString()
  });
});

// Global error handler
app.use((error, req, res, next) => {
  console.error('Global Error Handler:', error);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    code: 'INTERNAL_SERVER_ERROR'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`Node.js server running on port ${PORT}`);
  console.log(`Python API endpoint: ${config.PYTHON_API_URL}`);
});