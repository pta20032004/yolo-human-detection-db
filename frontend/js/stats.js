/**
 * stats.js
 * Handle statistics tracking (FPS, counts, emotions, recognition)
 */

import state from './state.js';
import config from './config.js';

// DOM elements
let personCountElement, faceCountElement, recognizedCountElement, fpsCounterElement, emotionsStatsElement;

/**
 * Initialize stats module
 * @param {Object} elements - DOM elements
 */
export function initStats(elements) {
    personCountElement = elements.personCountElement;
    faceCountElement = elements.faceCountElement;
    recognizedCountElement = elements.recognizedCountElement;
    fpsCounterElement = elements.fpsCounterElement;
    emotionsStatsElement = document.getElementById('emotionsStats');
}

/**
 * Setup FPS counter
 */
export function setupFpsCounter() {
    // Clear existing timer if any
    if (state.fpsTimerId) {
        clearInterval(state.fpsTimerId);
    }
    
    // Setup new timer
    state.frameCount = 0;
    state.lastFrameTime = performance.now();
    
    state.fpsTimerId = setInterval(() => {
        const currentTime = performance.now();
        const elapsedTime = (currentTime - state.lastFrameTime) / 1000;
        
        if (elapsedTime > 0) {
            const fps = Math.round(state.frameCount / elapsedTime);
            fpsCounterElement.textContent = fps;
            
            // Reset counters
            state.frameCount = 0;
            state.lastFrameTime = currentTime;
        }
    }, 1000);
    
    console.log("FPS counter started");
}

/**
 * Update statistics display
 * @param {number} personCount - Number of detected persons
 * @param {number} faceCount - Number of detected faces
 * @param {Array} faceBoxes - Face detection boxes with emotion and recognition data
 */
export function updateStats(personCount, faceCount, faceBoxes) {
    personCountElement.textContent = personCount;
    faceCountElement.textContent = faceCount;
    
    // Count only recognized faces (not Unknown)
    let recognizedFaces = 0;
    const emotionCounts = {};
    
    if (faceBoxes && faceBoxes.length > 0) {
        faceBoxes.forEach(box => {
            // Count recognized faces
            if (box.recognition && box.recognition.is_known && box.recognition.name !== 'Unknown') {
                recognizedFaces++;
            }
            
            // Count emotions
            if (box.emotion) {
                emotionCounts[box.emotion] = (emotionCounts[box.emotion] || 0) + 1;
            }
        });
    }
    
    // Update recognized count
    recognizedCountElement.textContent = recognizedFaces;
    
    // Update emotion statistics
    updateEmotionStats(emotionCounts);
}

/**
 * Update emotion statistics
 * @param {Object} emotionCounts - Emotion counts object
 */
function updateEmotionStats(emotionCounts) {
    // Skip if element doesn't exist yet
    if (!emotionsStatsElement) return;
    
    // Build HTML for emotion stats
    let statsHtml = '';
    
    // Emotion statistics
    if (config.showEmotions) {
        statsHtml += '<h4>Cảm xúc:</h4>';
        
        if (Object.keys(emotionCounts).length === 0) {
            statsHtml += '<div class="emotion-stat-item">Không có dữ liệu</div>';
        } else {
            Object.entries(emotionCounts).forEach(([emotion, count]) => {
                const color = config.emotionColors[emotion] || config.emotionColor;
                statsHtml += `
                    <div class="emotion-stat-item">
                        <span class="emotion-indicator" style="background-color: ${color}"></span>
                        <span class="emotion-name">${emotion}:</span>
                        <span class="emotion-count">${count}</span>
                    </div>
                `;
            });
        }
    }
    
    // Update the DOM
    emotionsStatsElement.innerHTML = statsHtml;
}

/**
 * Reset statistics counters
 */
export function resetStats() {
    personCountElement.textContent = '0';
    faceCountElement.textContent = '0';
    recognizedCountElement.textContent = '0';
    fpsCounterElement.textContent = '0';
    
    // Clear stats
    if (emotionsStatsElement) {
        emotionsStatsElement.innerHTML = '';
    }
}