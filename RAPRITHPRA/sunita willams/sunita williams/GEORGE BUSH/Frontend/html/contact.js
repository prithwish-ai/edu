// Handle Contact Form Submission
document.getElementById("contactForm").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent form submission
  
    // Get form values
    const name = document.getElementById("name").value;
    const email = document.getElementById("email").value;
    const message = document.getElementById("message").value;
  
    // Basic validation
    if (!name || !email || !message) {
      alert("Please fill in all fields.");
      return;
    }
  
    // Simulate form submission (replace with actual API call)
    console.log("Form submitted:", { name, email, message });
  
    // Show success message
    alert("Thank you for contacting us! We'll get back to you soon.");
    document.getElementById("contactForm").reset(); // Clear the form
  });