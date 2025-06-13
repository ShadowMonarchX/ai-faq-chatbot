let token = "";

function login() {
  const username = document.getElementById("username").value;
  const password = document.getElementById("password").value;

  fetch("/auth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `username=${username}&password=${password}`
  })
    .then(res => res.json())
    .then(data => {
      token = data.access_token;
      localStorage.setItem("token", token);
      alert("Login successful!");
    })
    .catch(err => alert("Login failed"));
}

function sendQuery() {
  const question = document.getElementById("question").value;
  const token = localStorage.getItem("token");

  fetch("/chat", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/x-www-form-urlencoded"
    },
    body: `query=${encodeURIComponent(question)}`
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("response").innerText = "🤖 " + data.answer;
    })
    .catch(err => {
      document.getElementById("response").innerText = "❌ Error";
    });
}
