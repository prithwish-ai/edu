// signup.js

// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
  const form = document.querySelector('#signupForm');
  const inputs = form.querySelectorAll('input');
  const submitBtn = form.querySelector('.btn');
  const darkModeToggle = document.querySelector('.dark-mode-toggle');
  const socialButtons = document.querySelectorAll('.social-btn');

  // Form Validation
  function validateForm() {
    let isValid = true;
    const name = form.querySelector('input[name="name"]');
    const email = form.querySelector('input[name="email"]');
    const password = form.querySelector('input[name="password"]');
    const confirmPassword = form.querySelector('input[name="confirm-password"]');

    // Reset error messages
    form.querySelectorAll('.error-message').forEach(error => error.textContent = '');

    // Name validation
    if (name.value.trim().length < 2) {
      showError(name, 'Name must be at least 2 characters long');
      isValid = false;
    }

    // Email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email.value)) {
      showError(email, 'Please enter a valid email address');
      isValid = false;
    }

    // Password validation
    if (password.value.length < 8) {
      showError(password, 'Password must be at least 8 characters long');
      isValid = false;
    }

    // Confirm password validation
    if (password.value !== confirmPassword.value) {
      showError(confirmPassword, 'Passwords do not match');
      isValid = false;
    }

    return isValid;
  }

  // Show error message
  function showError(input, message) {
    const formGroup = input.closest('.form-group');
    let error = formGroup.querySelector('.error-message');
    if (error) {
      error.textContent = message;
    }
  }

  // Password strength meter
  const passwordInput = form.querySelector('input[name="password"]');
  if (passwordInput) {
    const strengthMeter = document.createElement('div');
    strengthMeter.className = 'password-strength';
    passwordInput.parentNode.appendChild(strengthMeter);

    passwordInput.addEventListener('input', () => {
      const password = passwordInput.value;
      let strength = 0;
      
      if (password.length >= 8) strength++;
      if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength++;
      if (password.match(/\d/)) strength++;
      if (password.match(/[^a-zA-Z\d]/)) strength++;
      
      strengthMeter.innerHTML = `
        <div class="strength-bar" style="width: ${(strength / 4) * 100}%"></div>
        <div class="strength-text">${getStrengthText(strength)}</div>
      `;
    });
  }

  function getStrengthText(strength) {
    const texts = ['Weak', 'Medium', 'Good', 'Strong'];
    return texts[strength - 1] || 'Weak';
  }

  // Password visibility toggle
  const passwordInputs = form.querySelectorAll('input[type="password"]');
  passwordInputs.forEach(input => {
    const toggle = document.createElement('span');
    toggle.className = 'password-toggle';
    toggle.innerHTML = '<i class="fas fa-eye"></i>';
    input.parentNode.appendChild(toggle);
    
    toggle.addEventListener('click', () => {
      const type = input.type === 'password' ? 'text' : 'password';
      input.type = type;
      toggle.innerHTML = `<i class="fas fa-eye${type === 'password' ? '' : '-slash'}"></i>`;
    });
  });

  // Form submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    // Show loading state
    submitBtn.disabled = true;
    const btnText = submitBtn.innerHTML;
    submitBtn.innerHTML = '<span class="btn-loader"></span>';

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Show success message
      showNotification('Account created successfully!', 'success');
      
      // Redirect to login page after 2 seconds
      setTimeout(() => {
        window.location.href = 'login.html';
      }, 2000);
    } catch (error) {
      showNotification('An error occurred. Please try again.', 'error');
      submitBtn.disabled = false;
      submitBtn.innerHTML = btnText;
    }
  });

  // Input validation on blur
  inputs.forEach(input => {
    input.addEventListener('blur', () => {
      validateForm();
    });
  });

  // Dark mode toggle
  const isDarkMode = localStorage.getItem('darkMode') === 'true';
  if (isDarkMode) document.body.classList.add('dark-mode');

  darkModeToggle.addEventListener('click', () => {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
  });

  // Social login buttons
  socialButtons.forEach(button => {
    button.addEventListener('click', () => {
      const provider = button.classList[1];
      showNotification(`Connecting to ${provider}...`, 'info');
      
      // Simulate social login
      setTimeout(() => {
        showNotification(`${provider} login successful!`, 'success');
      }, 1500);
    });
  });

  // Remember me checkbox
  const rememberMe = form.querySelector('input[name="remember"]');
  if (rememberMe) {
    rememberMe.addEventListener('change', () => {
      if (rememberMe.checked) {
        showNotification('Your preferences will be remembered', 'info');
      }
    });
  }

  // Terms and Privacy Policy
  const termsCheckbox = form.querySelector('input[name="terms"]');
  if (termsCheckbox) {
    termsCheckbox.addEventListener('change', () => {
      if (!termsCheckbox.checked) {
        showError(termsCheckbox, 'You must accept the terms and privacy policy');
      } else {
        const error = termsCheckbox.closest('.form-group').querySelector('.error-message');
        if (error) error.textContent = '';
      }
    });
  }
});

// Notification system
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.textContent = message;
  
  document.body.appendChild(notification);
  
  // Trigger animation
  setTimeout(() => notification.classList.add('show'), 100);
  
  // Remove notification after 3 seconds
  setTimeout(() => {
    notification.classList.remove('show');
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}