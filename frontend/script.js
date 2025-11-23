document.addEventListener('DOMContentLoaded', () => {
    const queryInput = document.getElementById('query-input');
    const submitButton = document.getElementById('submit-button');
    const chatContainer = document.getElementById('chat-container');
    const welcomeView = document.getElementById('welcome-view');
    const promptCards = document.querySelectorAll('.prompt-card');

    // const API_URL = 'http://127.0.0.1:8000/backend/query';
    const API_URL = '/backend/query';
    let chatHistory = [];

    // Auto-resize textarea
    queryInput.addEventListener('input', () => {
        queryInput.style.height = 'auto';
        queryInput.style.height = (queryInput.scrollHeight) + 'px';
        submitButton.disabled = queryInput.value.trim() === '';
    });

    // Handle form submission
    queryInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleQuery();
        }
    });

    submitButton.addEventListener('click', handleQuery);

    // Prompt card handlers
    promptCards.forEach(card => {
        card.addEventListener('click', () => {
            const text = card.querySelector('.card-text').textContent;
            queryInput.value = text;
            queryInput.dispatchEvent(new Event('input'));
            queryInput.focus();
        });
    });

    async function handleQuery() {
        const query = queryInput.value.trim();
        if (!query) return;
        
        // Hide welcome view if shown
        if (welcomeView && welcomeView.style.display !== 'none') {
            welcomeView.style.display = 'none';
            chatContainer.classList.remove('d-none');
        }
        
        // Add user message
        addMessage(query, 'user');
        chatHistory.push({ role: 'user', content: query });
        
        // Clear input
        queryInput.value = '';
        queryInput.style.height = 'auto';
        submitButton.disabled = true;
        queryInput.disabled = true;

        // Create loading message
        const loadingDots = createLoadingDots();
        const aiMessageElement = addMessage(loadingDots.outerHTML, 'ai');
        const aiMessageIndex = chatHistory.push({ role: 'ai', content: '' }) - 1;

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query,
                    history: chatHistory.slice(0, -1)
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();
            updateMessage(aiMessageElement, marked.parse(data.answer));
            chatHistory[aiMessageIndex].content = data.answer;
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = `**Error:** I encountered an issue processing your request.<br><br>*Details: ${error.message}*`;
            updateMessage(aiMessageElement, marked.parse(errorMessage));
            chatHistory[aiMessageIndex].content = errorMessage;
        } finally {
            queryInput.disabled = false;
            queryInput.focus();
        }
    }

    function createLoadingDots() {
        const container = document.createElement('div');
        container.className = 'loading-container';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'loading-dot';
            container.appendChild(dot);
        }
        
        return container;
    }

    function addMessage(content, role) {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `chat-message ${role}`;
        
        const avatarIcon = role === 'user' ? 'bi-person' : 'bi-graph-up-arrow';
        const avatar = `<div class="avatar"><i class="bi ${avatarIcon}"></i></div>`;
        
        const messageContent = `<div class="message-content">${content}</div>`;
        
        messageWrapper.innerHTML = avatar + messageContent;
        chatContainer.appendChild(messageWrapper);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return messageWrapper;
    }
    
    function updateMessage(messageElement, newContent) {
        const contentDiv = messageElement.querySelector('.message-content');
        if (contentDiv) {
            contentDiv.innerHTML = newContent;
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
});