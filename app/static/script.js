document.getElementById("recommend-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const customer_id = document.getElementById("customer_id").value;
  const top_n = parseInt(document.getElementById("top_n").value);

  const response = await fetch("/user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id, top_n })
  });

  const resultDiv = document.getElementById("results");
  if (response.ok) {
    const data = await response.json();
    resultDiv.innerHTML = `<h3>Recommended Products:</h3><ul>${data.map(item => `<li>${item}</li>`).join("")}</ul>`;
  } else {
    const error = await response.json();
    resultDiv.innerHTML = `<p style="color:red;">Error: ${error.detail}</p>`;
  }
});
