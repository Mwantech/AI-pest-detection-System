CREATE DATABASE pest_detection;
-- Store pest detection submissions
CREATE TABLE pest_detections (
    detection_id SERIAL PRIMARY KEY,
    image_path VARCHAR(255) NOT NULL,
    submission_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    device_info VARCHAR(255)
);

-- Store prediction results for each detection
CREATE TABLE prediction_results (
    prediction_id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES pest_detections(detection_id),
    pest_type VARCHAR(100) NOT NULL,
    confidence_score DECIMAL(5,2) NOT NULL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Store actions taken based on predictions
CREATE TABLE detection_actions (
    action_id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES pest_detections(detection_id),
    action_taken VARCHAR(255),
    action_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_status VARCHAR(50),
    notes TEXT
);

-- Store user feedback on predictions (optional)
CREATE TABLE prediction_feedback (
    feedback_id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES pest_detections(detection_id),
    correct_prediction BOOLEAN,
    actual_pest_type VARCHAR(100),
    feedback_notes TEXT,
    feedback_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX idx_pest_detections_date ON pest_detections(submission_date);
CREATE INDEX idx_prediction_results_pest ON prediction_results(pest_type);
CREATE INDEX idx_prediction_results_detection ON prediction_results(detection_id);