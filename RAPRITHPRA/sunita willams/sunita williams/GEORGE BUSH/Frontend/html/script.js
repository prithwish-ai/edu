// Chatbot Functionality
const chatbotToggle = document.getElementById("chatbotToggle");
const chatbotContainer = document.querySelector(".chatbot-container");
const closeChatbot = document.getElementById("closeChatbot");
const chatbotMessages = document.getElementById("chatbotMessages");
const chatbotInput = document.getElementById("chatbotInput");
const sendMessageButton = document.getElementById("sendMessage");

// Toggle Chatbot
chatbotToggle.addEventListener("click", () => {
  chatbotContainer.classList.toggle("active");
});

// Close Chatbot
closeChatbot.addEventListener("click", () => {
  chatbotContainer.classList.remove("active");
});

// Send Message
sendMessageButton.addEventListener("click", () => {
  const userMessage = chatbotInput.value.trim();
  if (userMessage) {
    // Add user message
    chatbotMessages.innerHTML += `
      <div class="chatbot-message user">
        <p>${userMessage}</p>
      </div>
    `;
    chatbotInput.value = "";

    // Simulate bot response
    setTimeout(() => {
      chatbotMessages.innerHTML += `
        <div class="chatbot-message bot">
          <p>Thanks for your message! How can I assist you further?</p>
        </div>
      `;
      // Scroll to bottom
      chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }, 1000);
  }
});