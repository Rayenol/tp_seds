const params = new URLSearchParams(window.location.search);
const userId = params.get('user_id');

// Fetch and display user info
function fetchUserInfo() {
    fetch(`/users/${userId}`)
        .then(response => response.json())
        .then(user => {
            const userInfoDiv = document.getElementById('userInfo');
            userInfoDiv.innerHTML = `<p><strong>${user.name}</strong> (${user.email})</p>`;
        });
}

// Fetch and display posts
function fetchPosts() {
    fetch(`/users/${userId}/posts/`)
        .then(response => response.json())
        .then(posts => {
            const postsDiv = document.getElementById('posts');
            postsDiv.innerHTML = ''; // Clear previous content
            posts.forEach(post => {
                const postElement = document.createElement('div');
                postElement.innerHTML = `
                    <p><strong>${post.title}</strong>: ${post.content}</p>
                `;
                postsDiv.appendChild(postElement);
            });
        });
}

// Add a new post
document.getElementById('addPostForm').addEventListener('submit', event => {
    event.preventDefault();
    const formData = new FormData(event.target);

    fetch(`/users/${userId}/posts/new`, {
        method: 'POST',
        body: JSON.stringify({
            title: formData.get('title'),
            content: formData.get('content'),
        }),
        headers: {
            'Content-Type': 'application/json',
        },
    }).then(response => {
        if (response.ok) {
            alert('Post added successfully!');
            fetchPosts(); // Refresh post list
        } else {
            alert('Failed to add post.');
        }
    });
}

// Initial fetch
fetchUserInfo();
fetchPosts();
