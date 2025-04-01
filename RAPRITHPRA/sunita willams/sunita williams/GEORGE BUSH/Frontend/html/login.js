// login.js

document.addEventListener('DOMContentLoaded', function () {
  const loginForm = document.getElementById('loginForm');
  const emailInput = document.getElementById('email');
  const passwordInput = document.getElementById('password');
  const emailError = document.getElementById('emailError');
  const passwordError = document.getElementById('passwordError');
  const btn = document.querySelector('.btn');
  const btnText = document.querySelector('.btn-text');
  const btnLoader = document.querySelector('.btn-loader');
  const darkModeToggle = document.querySelector('.dark-mode-toggle');

  // Dark Mode Toggle
  darkModeToggle.addEventListener('click', function () {
    document.body.classList.toggle('dark-mode');
  });

  // Form Submission
  loginForm.addEventListener('submit', function (e) {
    e.preventDefault();

    // Validate Email
    if (!emailInput.value.includes('@')) {
      emailError.textContent = 'Please enter a valid email address.';
      emailInput.classList.add('error');
    } else {
      emailError.textContent = '';
      emailInput.classList.remove('error');
    }

    // Validate Password
    if (passwordInput.value.length < 6) {
      passwordError.textContent = 'Password must be at least 6 characters.';
      passwordInput.classList.add('error');
    } else {
      passwordError.textContent = '';
      passwordInput.classList.remove('error');
    }

    // If no errors, simulate login
    if (emailError.textContent === '' && passwordError.textContent === '') {
      btn.classList.add('loading');
      btnText.style.opacity = '0';
      btnLoader.style.opacity = '1';

      setTimeout(() => {
        btn.classList.remove('loading');
        btnText.style.opacity = '1';
        btnLoader.style.opacity = '0';
        alert('Login successful! Redirecting...');
        window.location.href = 'dashboard.html'; // Redirect to dashboard
      }, 2000);
    }
  });
});