// books.js

document.addEventListener('DOMContentLoaded', function() {
    // Dark mode toggle functionality
    const darkModeToggle = document.getElementById('darkModeToggle');
    const body = document.body;

    // Check for saved user preference
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode === 'enabled') {
        body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
    }

    // Toggle dark mode
    darkModeToggle.addEventListener('click', function() {
        if (body.classList.contains('dark-mode')) {
            body.classList.remove('dark-mode');
            localStorage.setItem('darkMode', 'disabled');
            darkModeToggle.innerHTML = '<i class="fas fa-moon"></i><span>Dark Mode</span>';
        } else {
            body.classList.add('dark-mode');
            localStorage.setItem('darkMode', 'enabled');
            darkModeToggle.innerHTML = '<i class="fas fa-sun"></i><span>Light Mode</span>';
        }
    });

    // Search functionality
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');
    const bookGrid = document.getElementById('bookGrid');
    const loader = document.getElementById('loader');

    // Sample agricultural and technical books for simulation
    const agriculturalBooks = [
        {
            title: 'Modern Farming Techniques',
            author: 'Dr. Rajesh Kumar',
            description: 'A comprehensive guide to sustainable agriculture and modern farming methods.',
            image: 'https://images.unsplash.com/photo-1571504211935-1c936b327411?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Irrigation Systems & Water Management',
            author: 'Prof. Amit Sharma',
            description: 'Learn about advanced irrigation techniques for optimal crop growth.',
            image: 'https://images.unsplash.com/photo-1589923188900-85dae523342b?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Crop Disease Management',
            author: 'Dr. Priya Patel',
            description: 'Identify and treat common crop diseases to improve yield and quality.',
            image: 'https://images.unsplash.com/photo-1592982537447-7440770cbfc9?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Electrical Engineering for ITI',
            author: 'Prof. Vikram Singh',
            description: 'A practical approach to electrical engineering fundamentals for ITI students.',
            image: 'https://images.unsplash.com/photo-1625246333195-78d9c38ad449?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Soil Science & Fertility Management',
            author: 'Dr. Sanjay Mishra',
            description: 'Understanding soil composition and fertility for improved agricultural outcomes.',
            image: 'https://images.unsplash.com/photo-1464226184884-fa280b87c399?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Farm Equipment Maintenance',
            author: 'Eng. Ravi Kapoor',
            description: 'Learn to maintain and repair common agricultural machinery and equipment.',
            image: 'https://images.unsplash.com/photo-1520453803296-c39eabe2dab4?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Organic Farming Principles',
            author: 'Dr. Neha Joshi',
            description: 'A guide to implementing organic farming principles for sustainable agriculture.',
            image: 'https://images.unsplash.com/photo-1500651230702-0e2d8a49d4ad?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        },
        {
            title: 'Welding & Fabrication Techniques',
            author: 'Eng. Manoj Kumar',
            description: 'Master essential welding and metal fabrication skills with practical guidance.',
            image: 'https://images.unsplash.com/photo-1605000797499-95a51c5269ae?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=80'
        }
    ];

    // Function to display books
    function displayBooks(books) {
        bookGrid.innerHTML = '';
        
        if (books.length === 0) {
            bookGrid.innerHTML = '<div class="no-results">No books found. Try a different search term.</div>';
            return;
        }
        
        books.forEach(book => {
            const bookCard = document.createElement('div');
            bookCard.className = 'book-card';
            
            bookCard.innerHTML = `
                <img src="${book.image}" alt="${book.title}">
                <h3>${book.title}</h3>
                <p>${book.description}</p>
                <p class="author"><small>By ${book.author}</small></p>
            `;
            
            bookGrid.appendChild(bookCard);
        });
    }

    // Search function
    function searchBooks(query) {
        loader.style.display = 'flex';
        
        // Simulate API delay
        setTimeout(() => {
            const filteredBooks = agriculturalBooks.filter(book => {
                const searchTerm = query.toLowerCase();
                return (
                    book.title.toLowerCase().includes(searchTerm) ||
                    book.author.toLowerCase().includes(searchTerm) ||
                    book.description.toLowerCase().includes(searchTerm)
                );
            });
            
            displayBooks(filteredBooks);
            loader.style.display = 'none';
        }, 1000);
    }

    // Search event listeners
    searchButton.addEventListener('click', function() {
        searchBooks(searchInput.value);
    });
    
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchBooks(searchInput.value);
        }
    });

    // Add author class to initial cards for styling
    const initialBookCards = document.querySelectorAll('.book-card');
    initialBookCards.forEach(card => {
        const title = card.querySelector('h3').textContent;
        const matchingBook = agriculturalBooks.find(book => book.title === title);
        if (matchingBook) {
            const authorPara = document.createElement('p');
            authorPara.className = 'author';
            authorPara.innerHTML = `<small>By ${matchingBook.author}</small>`;
            card.appendChild(authorPara);
        }
    });
});