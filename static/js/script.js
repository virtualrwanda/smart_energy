document.addEventListener('DOMContentLoaded', () => {
    // --- Register Meter Form Logic ---
    const registerForm = document.getElementById('registerMeterForm');
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const serialNumber = document.getElementById('serialNumber').value;
            const ownerName = document.getElementById('ownerName').value;
            const ownerContact = document.getElementById('ownerContact').value;
            const messageDiv = document.getElementById('message');

            try {
                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        serial_number: serialNumber,
                        owner_name: ownerName,
                        owner_contact: ownerContact,
                    }),
                });

                const data = await response.json();
                if (response.ok) {
                    messageDiv.textContent = data.message;
                    messageDiv.className = 'message success';
                    registerForm.reset();
                } else {
                    messageDiv.textContent = data.error || 'An error occurred during registration.';
                    messageDiv.className = 'message error';
                }
            } catch (error) {
                console.error('Error:', error);
                messageDiv.textContent = 'Network error or server unavailable.';
                messageDiv.className = 'message error';
            }
        });
    }

    // --- Recharge Meter Form Logic ---
    const rechargeForm = document.getElementById('rechargeMeterForm');
    if (rechargeForm) {
        rechargeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const serial_number = document.getElementById('serial_number').value;
            const rechargeAmount = document.getElementById('rechargeAmount').value;
            const messageDiv = document.getElementById('message');

            try {
                const response = await fetch('/api/recharge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        meter_serial_number: serial_number,
                        recharge_amount: parseFloat(rechargeAmount),
                    }),
                });

                const data = await response.json();
                if (response.ok) {
                    messageDiv.textContent = data.message + ` New balance: ${data.new_balance} kWh`;
                    messageDiv.className = 'message success';
                    rechargeForm.reset();
                } else {
                    messageDiv.textContent = data.error || 'An error occurred during recharge.';
                    messageDiv.className = 'message error';
                }
            } catch (error) {
                console.error('Error:', error);
                messageDiv.textContent = 'Network error or server unavailable.';
                messageDiv.className = 'message error';
            }
        });
    }

    // --- Prediction Form Logic ---
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const serialNumber = document.getElementById('predictSerialNumber').value;
            const predictionResultDiv = document.getElementById('predictionResult');

            try {
                const response = await fetch('/predictx', { // Note: Changed to /predictx as per your backend
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ serial_number: serialNumber }),
                });

                const data = await response.json();
                if (response.ok) {
                    if (data.predicted_payment !== undefined) {
                        predictionResultDiv.textContent = `Predicted next payment: ${data.predicted_payment} RWF`;
                        predictionResultDiv.className = 'message success';
                    } else if (data.message) {
                        predictionResultDiv.textContent = data.message;
                        predictionResultDiv.className = 'message success';
                    } else {
                        predictionResultDiv.textContent = data.error || 'No prediction available.';
                        predictionResultDiv.className = 'message error';
                    }
                } else {
                    predictionResultDiv.textContent = data.error || 'Error getting prediction.';
                    predictionResultDiv.className = 'message error';
                }
            } catch (error) {
                console.error('Error:', error);
                predictionResultDiv.textContent = 'Network error or server unavailable.';
                predictionResultDiv.className = 'message error';
            }
        });
    }
});

// --- Fetch Registered Meters Function (for meter_list.html) ---
async function fetchRegisteredMeters() {
    const metersTableBody = document.getElementById('metersTableBody');
    const messageDiv = document.getElementById('message');
    if (!metersTableBody) return; // Exit if not on the correct page

    try {
        const response = await fetch('/api/registered_meters');
        const data = await response.json();

        if (response.ok) {
            metersTableBody.innerHTML = ''; // Clear existing rows
            if (data.length > 0) {
                data.forEach(meter => {
                    const row = metersTableBody.insertRow();
                    row.insertCell(0).textContent = meter[0]; // Serial Number
                    row.insertCell(1).textContent = meter[1]; // Owner Name
                    row.insertCell(2).textContent = meter[2]; // Owner Contact
                    row.insertCell(3).textContent = meter[3].toFixed(2); // Balance, formatted to 2 decimal places
                    row.insertCell(4).textContent = new Date(meter[4]).toLocaleString(); // Timestamp

                    const actionsCell = row.insertCell(5);
                    actionsCell.className = 'actions';

                    const historyButton = document.createElement('button');
                    historyButton.textContent = 'View History';
                    historyButton.onclick = () => window.location.href = `/history?serial_number=${meter[0]}`;
                    actionsCell.appendChild(historyButton);

                    const deleteButton = document.createElement('button');
                    deleteButton.textContent = 'Delete';
                    deleteButton.className = 'delete-btn';
                    deleteButton.onclick = () => deleteMeter(meter[0]);
                    actionsCell.appendChild(deleteButton);
                });
            } else {
                messageDiv.textContent = 'No registered meters found.';
                messageDiv.className = 'message';
            }
        } else {
            messageDiv.textContent = data.message || 'Error fetching registered meters.';
            messageDiv.className = 'message error';
        }
    } catch (error) {
        console.error('Error:', error);
        messageDiv.textContent = 'Network error or server unavailable.';
        messageDiv.className = 'message error';
    }
}

// --- Fetch Recharge History Function (for history.html) ---
async function fetchRechargeHistory(serialNumber) {
    const historyTableBody = document.getElementById('historyTableBody');
    const messageDiv = document.getElementById('message');
    if (!historyTableBody) return;

    try {
        const response = await fetch(`/api/recharges?serial_number=${serialNumber}`);
        const data = await response.json();

        if (response.ok) {
            historyTableBody.innerHTML = ''; // Clear existing rows
            if (data.length > 0) {
                data.forEach(recharge => {
                    const row = historyTableBody.insertRow();
                    row.insertCell(0).textContent = recharge[0].toFixed(2); // Recharge Amount
                    row.insertCell(1).textContent = new Date(recharge[1]).toLocaleString(); // Timestamp
                });
            } else {
                messageDiv.textContent = 'No recharge history found for this meter.';
                messageDiv.className = 'message';
            }
        } else {
            messageDiv.textContent = data.error || 'Error fetching recharge history.';
            messageDiv.className = 'message error';
        }
    } catch (error) {
        console.error('Error:', error);
        messageDiv.textContent = 'Network error or server unavailable.';
        messageDiv.className = 'message error';
    }
}

// --- Delete Meter Function ---
async function deleteMeter(serialNumber) {
    if (!confirm(`Are you sure you want to delete meter ${serialNumber}? This action cannot be undone.`)) {
        return;
    }

    const messageDiv = document.getElementById('message');
    try {
        const response = await fetch(`/api/delete/${serialNumber}`, {
            method: 'DELETE',
        });
        const data = await response.json();

        if (response.ok) {
            messageDiv.textContent = data.message;
            messageDiv.className = 'message success';
            fetchRegisteredMeters(); // Refresh the list after deletion
        } else {
            messageDiv.textContent = data.error || 'An error occurred during deletion.';
            messageDiv.className = 'message error';
        }
    } catch (error) {
        console.error('Error:', error);
        messageDiv.textContent = 'Network error or server unavailable.';
        messageDiv.className = 'message error';
    }
}
// Get CSRF token from meta tag
const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');

// Include in all fetch requests
fetch('/api/endpoint', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrfToken
    },
    body: JSON.stringify(data)
})