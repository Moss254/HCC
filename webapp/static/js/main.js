// Main JavaScript for HCC Prediction System

// Initialize tooltips
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    const tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // REMOVED: The generic button disabler code was causing the "Silent Click" issue.
    // We now let the specific pages (predict.html) handle their own buttons.
});

// Utility functions
const HCCUtils = {
    // Format percentage
    formatPercent: (value) => {
        return (value * 100).toFixed(1) + '%';
    },
    
    // Format risk level
    getRiskLevel: (probability) => {
        if (probability >= 0.7) return 'High';
        if (probability >= 0.3) return 'Medium';
        return 'Low';
    },
    
    // Validate email
    validateEmail: (email) => {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },
    
    // Download data as CSV
    downloadCSV: (data, filename) => {
        const csvContent = "data:text/csv;charset=utf-8," + data;
        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", filename);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
};

// API service
const HCCAPI = {
    baseURL: '',
    
    async request(endpoint, options = {}) {
        const url = this.baseURL + endpoint;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        try {
            const response = await fetch(url, config);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },
    
    // Prediction endpoints
    async predictSingle(data) {
        return this.request('/api/predict', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    async predictBatch(formData) {
        return this.request('/api/batch-predict', {
            method: 'POST',
            body: formData
        });
    },
    
    async getDashboardData() {
        return this.request('/api/dashboard-data');
    },
    
    async getFeatureConfig() {
        return this.request('/api/feature-config');
    }
};

// Chart utilities
const HCCCharts = {
    colors: {
        primary: '#667eea',
        success: '#27ae60',
        warning: '#f39c12',
        danger: '#e74c3c',
        info: '#3498db'
    },
    
    createPerformanceChart(ctx, data) {
        return new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
                datasets: [{
                    label: 'Model Performance',
                    data: data,
                    backgroundColor: 'rgba(102, 126, 234, 0.2)',
                    borderColor: '#667eea',
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#667eea'
                }]
            },
            options: {
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
    },
    
    createRiskDistributionChart(ctx, data) {
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High Risk', 'Medium Risk', 'Low Risk'],
                datasets: [{
                    data: data,
                    backgroundColor: [
                        this.colors.danger,
                        this.colors.warning,
                        this.colors.success
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { HCCUtils, HCCAPI, HCCCharts };
}