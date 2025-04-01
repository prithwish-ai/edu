// API Configuration
const API_URL = 'http://localhost:5500/api';

// DOM Elements
const chatContainer = document.getElementById('chat-container');
const landingContainer = document.getElementById('landing-container');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const fileUpload = document.getElementById('file-upload');
const filePreview = document.getElementById('file-preview');
const fileName = document.getElementById('file-name');
const fileSize = document.getElementById('file-size');
const removeFile = document.getElementById('remove-file');
const newChatBtn = document.getElementById('new-chat-btn');
const chatHistoryList = document.getElementById('chat-history-list');
const uploadPlantImage = document.getElementById('upload-plant-image');

// State variables
let selectedFile = null;
let chatHistory = [];
let currentChatId = null;
let serverConnected = false;

// Initialize the application
function init() {
    loadChatHistory();
    setupEventListeners();
    checkServerConnection();
}

// Check server connection
function checkServerConnection() {
    const connectionStatus = document.createElement('div');
    connectionStatus.style.position = 'fixed';
    connectionStatus.style.top = '10px';
    connectionStatus.style.right = '10px';
    connectionStatus.style.padding = '8px 12px';
    connectionStatus.style.borderRadius = '4px';
    connectionStatus.style.fontSize = '12px';
    connectionStatus.style.fontWeight = 'bold';
    connectionStatus.style.color = 'white';
    connectionStatus.style.backgroundColor = '#f44336';
    connectionStatus.textContent = 'Checking API connection...';
    document.body.appendChild(connectionStatus);
    
    // Check each API endpoint
    Promise.all([
        fetch(`${API_URL}/status`).catch(error => ({ ok: false, error })),
        fetch(`${API_URL}/plant-disease/status`).catch(error => ({ ok: false, error })),
        fetch(`${API_URL}/quiz/status`).catch(error => ({ ok: false, error })),
        fetch(`${API_URL}/study-materials/status`).catch(error => ({ ok: false, error })),
        fetch(`${API_URL}/progress/status`).catch(error => ({ ok: false, error })),
        fetch(`${API_URL}/topic-recommender/status`).catch(error => ({ ok: false, error }))
    ])
    .then(responses => {
        const connectedCount = responses.filter(res => res.ok).length;
        const totalCount = responses.length;
        
        if (connectedCount > 0) {
            serverConnected = true;
            connectionStatus.style.backgroundColor = '#4caf50';
            connectionStatus.textContent = `API Connected (${connectedCount}/${totalCount})`;
            
            // Fade out after 5 seconds
            setTimeout(() => {
                connectionStatus.style.transition = 'opacity 1s';
                connectionStatus.style.opacity = '0';
                setTimeout(() => connectionStatus.remove(), 1000);
            }, 5000);
        } else {
            connectionStatus.textContent = 'API Connection Failed';
            
            // Add retry button
            const retryButton = document.createElement('button');
            retryButton.textContent = 'Retry';
            retryButton.style.marginLeft = '8px';
            retryButton.style.padding = '2px 8px';
            retryButton.style.border = 'none';
            retryButton.style.borderRadius = '3px';
            retryButton.style.cursor = 'pointer';
            retryButton.addEventListener('click', () => {
                connectionStatus.textContent = 'Checking API connection...';
                connectionStatus.removeChild(retryButton);
                checkServerConnection();
            });
            connectionStatus.appendChild(retryButton);
        }
    });
}

// Set up event listeners
function setupEventListeners() {
    // Send message on button click
    sendButton.addEventListener('click', sendMessage);
    
    // Send message on Enter key (but not with Shift+Enter)
    chatInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // File upload handling
    fileUpload.addEventListener('change', handleFileUpload);
    removeFile.addEventListener('click', removeSelectedFile);
    
    // New chat button
    newChatBtn.addEventListener('click', startNewChat);
    
    // Example items
    document.querySelectorAll('.example-item').forEach(item => {
        item.addEventListener('click', function() {
            const query = this.getAttribute('data-query');
            chatInput.value = query;
            chatInput.focus();
        });
    });
    
    // Upload plant image option
    uploadPlantImage.addEventListener('click', function() {
        fileUpload.click();
    });
}

// Load chat history from local storage
function loadChatHistory() {
    const savedHistory = localStorage.getItem('chatHistory');
    if (savedHistory) {
        chatHistory = JSON.parse(savedHistory);
        updateChatHistoryList();
    }
}

// Save chat history to local storage
function saveChatHistory() {
    localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    updateChatHistoryList();
}

// Update the chat history sidebar
function updateChatHistoryList() {
    chatHistoryList.innerHTML = '';
    
    chatHistory.forEach(chat => {
        const chatItem = document.createElement('div');
        chatItem.className = 'chat-item';
        chatItem.textContent = chat.title || 'New conversation';
        chatItem.setAttribute('data-chat-id', chat.id);
        
        chatItem.addEventListener('click', function() {
            loadChat(chat.id);
        });
        
        chatHistoryList.appendChild(chatItem);
    });
}

// Start a new chat
function startNewChat() {
    // Generate a new chat ID
    currentChatId = Date.now().toString();
    
    // Create a new chat entry
    const newChat = {
        id: currentChatId,
        title: 'New conversation',
        messages: []
    };
    
    // Add to chat history
    chatHistory.unshift(newChat);
    saveChatHistory();
    
    // Clear the chat container and show landing view
    chatContainer.innerHTML = '';
    chatContainer.appendChild(landingContainer);
    
    // Clear input
    chatInput.value = '';
    removeSelectedFile();
}

// Load a specific chat
function loadChat(chatId) {
    const chat = chatHistory.find(c => c.id === chatId);
    
    if (chat) {
        currentChatId = chatId;
        
        // Clear the chat container
        chatContainer.innerHTML = '';
        
        // Display messages
        chat.messages.forEach(msg => {
            addMessageToUI(msg.role, msg.content, msg.additionalData);
        });
        
        // If no messages, show landing view
        if (chat.messages.length === 0) {
            chatContainer.appendChild(landingContainer);
        }
    }
}

// Handle file upload
function handleFileUpload(event) {
    if (event.target.files.length > 0) {
        selectedFile = event.target.files[0];
        
        // Update file preview
        fileName.textContent = selectedFile.name;
        fileSize.textContent = formatFileSize(selectedFile.size);
        filePreview.classList.add('active');
        
        // Show plant preview if it's an image
        if (selectedFile.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                // Check if a plant preview already exists and remove it
                const existingPreview = document.querySelector('.plant-image-preview');
                if (existingPreview) {
                    existingPreview.remove();
                }
                
                // Create new preview
                const imgPreview = document.createElement('img');
                imgPreview.src = e.target.result;
                imgPreview.className = 'plant-image-preview';
                filePreview.appendChild(imgPreview);
            };
            reader.readAsDataURL(selectedFile);
        }
    }
}

// Remove selected file
function removeSelectedFile() {
    selectedFile = null;
    fileUpload.value = '';
    filePreview.classList.remove('active');
    
    // Remove any plant image preview
    const plantPreview = document.querySelector('.plant-image-preview');
    if (plantPreview) {
        plantPreview.remove();
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    else if (bytes < 1048576) return (bytes / 1024).toFixed(2) + ' KB';
    else return (bytes / 1048576).toFixed(2) + ' MB';
}

// Send message to the chatbot
function sendMessage() {
    const message = chatInput.value.trim();
    
    // Don't send empty messages
    if (!message && !selectedFile) return;
    
    // Hide the landing view if visible
    if (landingContainer.parentNode) {
        landingContainer.remove();
    }
    
    // Add user message to UI
    addMessageToUI('user', message);
    
    // Clear input
    chatInput.value = '';
    
    // Add loading indicator
    const loadingElement = addLoadingIndicator();
    
    if (selectedFile && selectedFile.type.startsWith('image/')) {
        // Handle plant disease detection if an image is attached
        detectPlantDisease(selectedFile, message, loadingElement);
    } else {
        // Handle regular message
        sendChatMessage(message, loadingElement);
    }
    
    // Save the first message as the chat title if this is a new chat
    const currentChat = chatHistory.find(c => c.id === currentChatId);
    if (currentChat && currentChat.messages.length === 0) {
        currentChat.title = message.length > 30 ? message.substring(0, 30) + '...' : message;
        saveChatHistory();
    }
    
    // Add message to history
    addMessageToHistory('user', message);
}

// Send a chat message to the API
function sendChatMessage(message, loadingElement) {
    // API call to chatbot
    fetch(`${API_URL}/chat/message`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Remove loading indicator
        loadingElement.remove();
        
        // Add response to UI
        addMessageToUI('assistant', data.reply);
        
        // Add to history
        addMessageToHistory('assistant', data.reply);
    })
    .catch(error => {
        console.error('Error:', error);
        
        // Remove loading indicator
        loadingElement.remove();
        
        // Add error message
        addMessageToUI('assistant', 'Sorry, I encountered an error. Please try again later.');
        
        // Add to history
        addMessageToHistory('assistant', 'Sorry, I encountered an error. Please try again later.');
    });
}

// Detect plant disease from an image
function detectPlantDisease(file, message, loadingElement) {
    const formData = new FormData();
    formData.append('image', file);
    
    // Add any text message as context
    if (message) {
        formData.append('context', message);
    }
    
    // API call to plant disease detector
    fetch(`${API_URL}/plant-disease/detect`, {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Remove loading indicator
        loadingElement.remove();
        
        // Remove file preview
        removeSelectedFile();
        
        // Format and display the result
        const resultHtml = formatPlantDiseaseResult(data);
        
        // Add response to UI
        addMessageToUI('assistant', resultHtml, data);
        
        // Add to history
        addMessageToHistory('assistant', resultHtml, data);
    })
    .catch(error => {
        console.error('Error:', error);
        
        // Remove loading indicator
        loadingElement.remove();
        
        // Add error message
        addMessageToUI('assistant', 'Sorry, I encountered an error analyzing the plant image. Please try again later.');
        
        // Add to history
        addMessageToHistory('assistant', 'Sorry, I encountered an error analyzing the plant image. Please try again later.');
    });
}

// Format plant disease result
function formatPlantDiseaseResult(data) {
    const confidencePercent = Math.round(data.confidence * 100);
    
    return `
        <div class="plant-analysis-result">
            <h3>Plant Analysis Result</h3>
            <p><strong>Diagnosis:</strong> ${data.disease}</p>
            <p><strong>Confidence:</strong> ${confidencePercent}%</p>
            <div class="confidence-bar">
                <div class="confidence-level" style="width: ${confidencePercent}%"></div>
            </div>
            <div class="description">
                <p>${data.description}</p>
            </div>
            <div class="treatment-section">
                <h4>Recommended Treatment:</h4>
                <p>${data.treatment}</p>
            </div>
        </div>
    `;
}

// Add loading indicator
function addLoadingIndicator() {
    const loadingElement = document.createElement('div');
    loadingElement.className = 'message assistant loading';
    loadingElement.innerHTML = `
        <div class="message-avatar assistant-avatar">🌾</div>
        <div class="message-content">
            <div class="loading-spinner"></div>
            <span>Processing your request...</span>
        </div>
    `;
    
    chatContainer.appendChild(loadingElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return loadingElement;
}

// Add message to UI
function addMessageToUI(role, content, additionalData = null) {
    const messageElement = document.createElement('div');
    messageElement.className = `message ${role}`;
    
    const avatarElement = document.createElement('div');
    avatarElement.className = `message-avatar ${role}-avatar`;
    avatarElement.textContent = role === 'user' ? 'U' : '🌾';
    
    const contentElement = document.createElement('div');
    contentElement.className = 'message-content';
    contentElement.innerHTML = content;
    
    messageElement.appendChild(avatarElement);
    messageElement.appendChild(contentElement);
    
    // If there's an image URL in additionalData, add it
    if (additionalData && additionalData.image_url) {
        const imageElement = document.createElement('img');
        imageElement.src = additionalData.image_url;
        imageElement.className = 'plant-image-preview';
        contentElement.appendChild(imageElement);
    }
    
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Add message to history
function addMessageToHistory(role, content, additionalData = null) {
    const chat = chatHistory.find(c => c.id === currentChatId);
    
    if (chat) {
        chat.messages.push({
            role: role,
            content: content,
            additionalData: additionalData,
            timestamp: new Date().toISOString()
        });
        
        saveChatHistory();
    }
}

// Initialize when the page loads
window.addEventListener('DOMContentLoaded', init); 