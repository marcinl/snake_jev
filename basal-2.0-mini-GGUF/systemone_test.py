import requests

# 1. Define the basal-serve server endpoint
url = "http://127.0.0.1:8000/v1/systemone"

# 2. Structure your payload based on the basal formatting specification
payload = {
    "model": "basal-1.5-mini:q8_0",
    "state": "Customer: Since yesterday I cannot log in to my online banking. The system displays a password error.",
    "questions": {
        "routing_dept": {
            "type": "choice",
            "instructions": "Which department should receive this ticket?",
            "criteria": {
                "cards": "Card Complaints and Issues",
                "online_banking": "Electronic Banking Technical Support",
                "loans": "Loans and Mortgages Department"
            }
        }
    }
}

# 3. Send the POST request to the server
headers = {"Content-Type": "application/json"}
try:
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()  # Check for HTTP request errors
    
    # 4. Print the server response
    print("Response Status:", response.status_code)
    print("Result:", response.json())

except requests.exceptions.RequestException as e:
    print(f"An error occurred connecting to basal-serve: {e}")
