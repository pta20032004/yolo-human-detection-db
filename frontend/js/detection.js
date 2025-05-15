/**
 * detection.js
 * Handle frame processing and drawing detection results
 */

import state from './state.js';
import config from './config.js';
import { updateStats, setupFpsCounter } from './stats.js';

// DOM elements
let video, overlay, ctx;

/**
 * Initialize detection module
 * @param {Object} elements - DOM elements
 */
export function initDetection(elements) {
    video = elements.video;
    overlay = elements.overlay;
    ctx = overlay.getContext('2d');
}

/**
 * Start frame processing
 */
export function startProcessing() {
    if (state.isProcessing) return;
    
    state.isProcessing = true;
    
    // Setup FPS counter
    setupFpsCounter();
    
    // Start processing frames
    processFrame();
}

/**
 * Stop frame processing
 */
export function stopProcessing() {
    state.isProcessing = false;
    
    // Cancel pending processing timer
    if (state.processingTimerId) {
        clearTimeout(state.processingTimerId);
        state.processingTimerId = null;
    }
}

/**
 * Process a single frame and schedule the next one
 */
export async function processFrame() {
    // Check if processing is still active
    if (!state.isProcessing || !state.isRunning) {
        return;
    }
    
    const startTime = performance.now();
    
    try {
        // Capture frame from video
        const canvas = document.createElement('canvas');
        const tempCtx = canvas.getContext('2d');
        
        // Check if video is ready
        if (video.videoWidth === 0 || video.videoHeight === 0) {
            // Video not ready, retry later
            state.processingTimerId = setTimeout(processFrame, 100);
            return;
        }
        
        // Set canvas dimensions to match video
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Draw current frame to temporary canvas
        tempCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg', 0.8);
        });
        
        // Prepare data to send
        const formData = new FormData();
        formData.append('file', blob, 'frame.jpg');
        
        // Send frame to server for processing
        const response = await fetch(config.serverUrl, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        // Process server response
        const result = await response.json();
        
        // Save results to application state
        state.personBoxes = result.person_boxes || [];
        state.faceBoxes = result.face_boxes || [];
        
        // Draw detection results
        drawDetections(state.personBoxes, state.faceBoxes);
        
        // Update statistics
        updateStats(result.persons, result.faces, state.faceBoxes);
        
        // Increment frame counter for FPS calculation
        state.frameCount++;
        
        // Calculate frame processing time
        const processingTime = performance.now() - startTime;
        
        // Calculate delay for desired frame rate
        const targetFrameTime = 1000 / config.frameRate;
        const delayTime = Math.max(0, targetFrameTime - processingTime);
        
        // Schedule next frame if application is still running
        if (state.isProcessing && state.isRunning) {
            state.processingTimerId = setTimeout(processFrame, delayTime);
        }
        
    } catch (error) {
        console.error('Lỗi khi xử lý frame:', error);
        
        // If error occurs, retry after 1 second if application is still running
        if (state.isProcessing && state.isRunning) {
            state.processingTimerId = setTimeout(processFrame, 1000);
        }
    }
}

/**
 * Draw detection boxes for persons and faces
 * @param {Array} personBoxes - Person detection boxes
 * @param {Array} faceBoxes - Face detection boxes with emotion and recognition data
 */
export function drawDetections(personBoxes, faceBoxes) {
    // Only draw when application is running
    if (!state.isRunning) {
        return;
    }
    
    // Clear canvas before drawing
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    
    // Get font size from config
    const fontSize = config.isMobile ? 
                    (config.mobileLabelFontSize || 14) : 
                    (config.desktopLabelFontSize || 16);
    
    // Calculate label height and padding
    const labelPadding = config.labelPadding || 6;
    const labelMargin = config.labelMargin || 8;
    const labelHeight = fontSize + labelPadding * 2;
    const borderWidth = config.borderWidth || 4;
    
    // Draw person boxes if enabled
    if (config.showPersons && personBoxes && personBoxes.length > 0) {
        personBoxes.forEach(box => {
            const [x1, y1, x2, y2] = box.coords;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Draw person box
            ctx.strokeStyle = config.personColor;
            ctx.lineWidth = borderWidth;
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label if confidence display is enabled
            if (config.showConfidence) {
                // Set font size for label
                ctx.font = `bold ${fontSize}px Arial`;
                
                // Create label text
                const label = `Người ${box.confidence.toFixed(2)}`;
                const textWidth = ctx.measureText(label).width + labelPadding * 2;
                
                // Create label background
                ctx.fillStyle = config.personColor;
                ctx.fillRect(x1, y1 - labelHeight - labelMargin, textWidth, labelHeight);
                
                // Draw text
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(label, x1 + labelPadding, y1 - labelMargin - labelPadding);
            }
        });
    }
    
    // Draw face boxes if enabled
    if (config.showFaces && faceBoxes && faceBoxes.length > 0) {
        faceBoxes.forEach((box, index) => {
            // Check if coords exist and are valid
            if (!box.coords || box.coords.length < 4) {
                console.error('Invalid coords:', box.coords);
                return;
            }
            
            const [x1, y1, x2, y2] = box.coords;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Determine box color - chỉ dựa vào known/unknown
            let boxColor = config.faceColor; // Default green for faces
            if (box.recognition) {
                if (box.recognition.is_known && box.recognition.name && box.recognition.name !== 'Unknown') {
                    boxColor = config.knownFaceColor || '#4ecdc4'; // Blue for known
                } else {
                    boxColor = config.unknownFaceColor || '#ff6b6b'; // Red for unknown
                }
            }
            
            // Draw face box
            ctx.strokeStyle = boxColor;
            ctx.lineWidth = borderWidth;
            ctx.strokeRect(x1, y1, width, height);
            
            // Prepare labels
            let nameLabel = '';
            let emotionLabel = '';
            
            // Name label (hiển thị bên trái trên)
            if (box.recognition) {
                if (box.recognition.is_known && box.recognition.name && box.recognition.name !== 'Unknown') {
                    nameLabel = box.recognition.name;
                    if (config.showSimilarityScore && box.recognition.similarity) {
                        nameLabel += ` (${Math.round(box.recognition.similarity * 100)}%)`;
                    }
                } else {
                    nameLabel = 'Unknown';
                }
            } else if (config.showConfidence) {
                nameLabel = `Face ${box.confidence.toFixed(2)}`;
            }
            
            // Emotion label with confidence (hiển thị bên phải trên)
            if (config.showEmotions && box.emotion) {
                emotionLabel = box.emotion;
                
                // Thêm % confidence của emotion nếu có (từ emotion model)
                emotionLabel += ' (100%)'; // Placeholder - có thể implement emotion confidence nếu cần
            }
            
            // Set font for labels
            ctx.font = `bold ${fontSize}px Arial`;
            
            // Draw name label (bên trái trên face box)
            if (nameLabel) {
                const nameWidth = ctx.measureText(nameLabel).width + labelPadding * 2;
                
                // Name background
                ctx.fillStyle = boxColor;
                ctx.fillRect(x1, y1 - labelHeight - labelMargin, nameWidth, labelHeight);
                
                // Name text
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(nameLabel, x1 + labelPadding, y1 - labelMargin - labelPadding);
            }
            
            // Draw emotion label (bên phải trên face box)
            if (emotionLabel) {
                const emotionWidth = ctx.measureText(emotionLabel).width + labelPadding * 2;
                
                // Position emotion label to the right of the face box
                const emotionX = x2 - emotionWidth;
                
                // Use emotion-specific color if available
                const emotionColor = config.emotionColors[box.emotion] || config.emotionColor;
                
                // Emotion background
                ctx.fillStyle = emotionColor;
                ctx.fillRect(emotionX, y1 - labelHeight - labelMargin, emotionWidth, labelHeight);
                
                // Emotion text
                ctx.fillStyle = '#FFFFFF';
                ctx.fillText(emotionLabel, emotionX + labelPadding, y1 - labelMargin - labelPadding);
            }
        });
    }
}