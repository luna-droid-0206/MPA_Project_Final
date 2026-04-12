/**
 * SimCLR Web App - Main JavaScript
 * Provides common functionality across all pages
 */

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Format a number to a specified number of decimal places
 * @param {number} num - The number to format
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted number
 */
function formatNumber(num, decimals = 2) {
    return Number(num).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/**
 * Format bytes to human-readable format
 * @param {number} bytes - The number of bytes
 * @returns {string} Formatted file size
 */
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Show a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The type of toast ('success', 'error', 'info', 'warning')
 * @param {number} duration - Duration in milliseconds
 */
function showToast(message, type = 'info', duration = 3000) {
    const toastHtml = `
        <div class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-${type} text-white">
                <strong class="me-auto">
                    ${type.charAt(0).toUpperCase() + type.slice(1)}
                </strong>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;

    const toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'position-fixed bottom-0 end-0 p-3';
        container.style.zIndex = '11';
        document.body.appendChild(container);
    }

    const toastElement = document.createElement('div');
    toastElement.innerHTML = toastHtml;
    document.getElementById('toastContainer').appendChild(toastElement);

    const toast = new bootstrap.Toast(toastElement.querySelector('.toast'));
    toast.show();

    if (duration > 0) {
        setTimeout(() => {
            toastElement.remove();
        }, duration);
    }
}

/**
 * Validate file size
 * @param {File} file - The file to validate
 * @param {number} maxSize - Maximum file size in bytes
 * @returns {object} Validation result
 */
function validateFileSize(file, maxSize = 50 * 1024 * 1024) {
    return {
        valid: file.size <= maxSize,
        message: file.size > maxSize ? 
            `File size exceeds limit. Max: ${formatBytes(maxSize)}, Got: ${formatBytes(file.size)}` : 
            'File size is valid'
    };
}

/**
 * Validate image file
 * @param {File} file - The file to validate
 * @returns {object} Validation result
 */
function validateImageFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/webp'];
    const isValid = validTypes.includes(file.type);
    
    return {
        valid: isValid,
        message: isValid ? 'File type is valid' : `Invalid file type. Supported: ${validTypes.join(', ')}`
    };
}

// ============================================================================
// API Functions
// ============================================================================

/**
 * Fetch data from the API
 * @param {string} endpoint - The API endpoint
 * @param {object} options - Fetch options
 * @returns {Promise} Response data
 */
async function apiCall(endpoint, options = {}) {
    try {
        const response = await fetch(endpoint, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

/**
 * Get health status
 * @returns {Promise} Health status data
 */
async function getHealthStatus() {
    return apiCall('/health');
}

/**
 * Get model information
 * @returns {Promise} Model information
 */
async function getModelsInfo() {
    return apiCall('/api/models');
}

/**
 * Get evaluation results
 * @returns {Promise} Evaluation results
 */
async function getEvaluationResults() {
    return apiCall('/api/evaluation-results');
}

// ============================================================================
// DOM Utilities
// ============================================================================

/**
 * Show a loading state on an element
 * @param {Element|string} element - The element or selector
 * @param {boolean} show - Whether to show or hide
 */
function setLoading(element, show = true) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (!el) return;

    if (show) {
        el.disabled = true;
        el.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
            Loading...
        `;
        el.classList.add('disabled');
    } else {
        el.disabled = false;
        el.classList.remove('disabled');
    }
}

/**
 * Disable/enable multiple elements
 * @param {NodeList|Element[]} elements - Elements to toggle
 * @param {boolean} disabled - Whether to disable
 */
function setDisabled(elements, disabled = true) {
    const nodeList = elements instanceof NodeList ? elements : [elements];
    nodeList.forEach(el => {
        el.disabled = disabled;
        if (disabled) {
            el.classList.add('disabled');
        } else {
            el.classList.remove('disabled');
        }
    });
}

/**
 * Show/hide element
 * @param {Element|string} element - The element or selector
 * @param {boolean} show - Whether to show
 */
function toggleVisibility(element, show = true) {
    const el = typeof element === 'string' ? document.querySelector(element) : element;
    if (!el) return;
    el.style.display = show ? '' : 'none';
}

// ============================================================================
// Chart Utilities
// ============================================================================

/**
 * Create a bar chart
 * @param {string} canvasId - Canvas element ID
 * @param {string[]} labels - Chart labels
 * @param {number[]} data - Chart data
 * @param {object} options - Chart options
 * @returns {Chart} Chart instance
 */
function createBarChart(canvasId, labels, data, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: options.label || 'Data',
                data: data,
                backgroundColor: options.backgroundColor || '#0d6efd',
                borderRadius: options.borderRadius || 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: options.showLegend !== false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            ...options.scales
        }
    });
}

/**
 * Create a line chart
 * @param {string} canvasId - Canvas element ID
 * @param {string[]} labels - Chart labels
 * @param {number[]} data - Chart data
 * @param {object} options - Chart options
 * @returns {Chart} Chart instance
 */
function createLineChart(canvasId, labels, data, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: options.label || 'Data',
                data: data,
                borderColor: options.borderColor || '#0d6efd',
                backgroundColor: options.backgroundColor || 'rgba(13, 110, 253, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: options.showLegend !== false
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// ============================================================================
// Event Listeners
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips and popovers
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Log app initialization
    console.log('SimCLR Web App initialized');
});

// ============================================================================
// Export Functions
// ============================================================================

// Make functions available globally
window.appUtils = {
    formatNumber,
    formatBytes,
    showToast,
    validateFileSize,
    validateImageFile,
    apiCall,
    getHealthStatus,
    getModelsInfo,
    getEvaluationResults,
    setLoading,
    setDisabled,
    toggleVisibility,
    createBarChart,
    createLineChart
};
