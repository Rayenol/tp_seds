// Fetch and display all users
function fetchUsers() {
  fetch("/users/")
    .then((response) => response.json())
    .then((users) => {
      const usersDiv = document.getElementById("users");
      usersDiv.innerHTML = ""; // Clear previous content
      users.forEach((user) => {
        const userElement = document.createElement("div");
        userElement.innerHTML = `
                  <p>
                      <strong>${user.name}</strong> (${user.email})
                      <a href="/user_posts.html?user_id=${user.id}">View Posts</a>
                  </p>
              `;
        usersDiv.appendChild(userElement);
      });
    });
}

// Add a new user
document.getElementById("addUserForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const formData = new FormData(event.target);

  fetch("/users/new", {
    method: "POST",
    body: JSON.stringify({
      name: formData.get("name"),
      email: formData.get("email"),
    }),
    headers: {
      "Content-Type": "application/json",
    },
  }).then((response) => {
    if (response.ok) {
      alert("User added successfully!");
      fetchUsers(); // Refresh user list
    } else {
      alert("Failed to add user.");
    }
  });
});

// Initial fetch
fetchUsers();
