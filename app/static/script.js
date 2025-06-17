/**
 * Recommendation Form Submission Handler
 *
 * This script handles the submission of the recommendation form. It:
 * 1. Prevents the default form submission behavior.
 * 2. Retrieves user input (customer_id and number of top recommendations).
 * 3. Sends a POST request to the FastAPI backend (`/user`) with the input data.
 * 4. Parses the response and dynamically updates the HTML with the recommended products or an error message.
 */

// Attach an event listener to the form with ID "recommend-form"
document.getElementById("recommend-form").addEventListener("submit", async (e) => {
  e.preventDefault(); // Prevent page reload on form submission

  // Get user inputs from the form
  const customer_id = document.getElementById("customer_id").value; // User-entered Customer ID
  const top_n = parseInt(document.getElementById("top_n").value);   // Number of top recommendations

  // Send a POST request to the /user endpoint with the form data
  const response = await fetch("/user", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ customer_id, top_n })  // Convert data to JSON string
  });

  // Get the results container element to update it with response
  const resultDiv = document.getElementById("results");

  if (response.ok) {
    // If request is successful, parse and display recommended products
    const data = await response.json();  // Array of product names
    resultDiv.innerHTML = `<h3>Recommended Products:</h3><ul>${data.map(item => `<li>${item}</li>`).join("")}</ul>`;
  } else {
    // If request fails, show the error returned by the backend
    const error = await response.json();
    resultDiv.innerHTML = `<p style="color:red;">Error: ${error.detail}</p>`;
  }
});
