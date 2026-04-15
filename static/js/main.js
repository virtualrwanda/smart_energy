document.addEventListener('DOMContentLoaded', () => {
    // --- Helper Function to Get CSRF Token ---
    function getCsrfToken(form) {
        const tokenInput = form.querySelector('input[name="csrf_token"]');
        if (!tokenInput || !tokenInput.value) {
            console.error('CSRF token input not found or empty in form:', form.id);
            return null;
        }
        return tokenInput.value;
    }

    // --- Helper Function to Display Error/Success Messages ---
    function displayMessage(messageDiv, message, type) {
        messageDiv.textContent = message;
        messageDiv.className = `mt-4 p-3 rounded text-center flash-message ${type}`;
        messageDiv.classList.remove('hidden');
    }

    // --- Register Meter Form Logic ---
    const registerForm = document.getElementById('registerMeterForm');
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const serialNumber = document.getElementById('serialNumber').value.trim();
            const ownerName = document.getElementById('ownerName').value.trim();
            const ownerContact = document.getElementById('ownerContact').value.trim();
            const initialPassword = document.getElementById('initialPassword').value.trim();
            const messageDiv = document.getElementById('message');
            const submitBtn = document.getElementById('submitBtn');
            const csrfToken = getCsrfToken(registerForm);

            // Client-side validation
            if (!serialNumber || serialNumber.length < 3) {
                displayMessage(messageDiv, 'Meter serial number must be at least 3 characters long.', 'danger');
                return;
            }
            if (!ownerName || ownerName.length < 2) {
                displayMessage(messageDiv, 'Owner name must be at least 2 characters long.', 'danger');
                return;
            }
            if (!ownerContact || !/^[^\s@]+@[^\s@]+\.[^\s@]+$|^(\+\d{1,3})?\d{10}$/.test(ownerContact)) {
                displayMessage(messageDiv, 'Please enter a valid email or phone number (e.g., +1234567890 or user@example.com).', 'danger');
                return;
            }
            if (!csrfToken) {
                displayMessage(messageDiv, 'CSRF token is missing. Please refresh the page and try again.', 'danger');
                return;
            }

            messageDiv.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';

            try {
                const payload = {
                    serial_number: serialNumber,
                    owner_name: ownerName,
                    owner_contact: ownerContact
                };
                if (initialPassword) payload.initial_password = initialPassword;

                const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-Token': csrfToken
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (response.ok) {
                    displayMessage(messageDiv, `${data.message} Initial password: ${data.initial_password_set_to}`, 'success');
                    registerForm.reset();
                } else {
                    const errorMsg = data.details ? `${data.error}: ${data.details.map(e => e.msg || e.message).join(', ')}` : data.error || 'An error occurred during registration.';
                    displayMessage(messageDiv, errorMsg, 'danger');
                }
            } catch (error) {
                console.error('Registration failed:', error);
                displayMessage(messageDiv, 'Failed to register meter. Please check your connection and try again.', 'danger');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Register';
            }
        });
    }

    // --- Recharge Meter Form Logic ---
    const rechargeForm = document.getElementById('rechargeMeterForm');
    if (rechargeForm) {
        rechargeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const serialNumber = document.getElementById('serial_number').value.trim();
            const rechargeAmount = parseFloat(document.getElementById('rechargeAmount').value);
            const messageDiv = document.getElementById('message');
            const submitBtn = document.getElementById('submitBtn');
            const csrfToken = getCsrfToken(rechargeForm);

            if (!serialNumber) {
                displayMessage(messageDiv, 'Please enter a valid meter serial number.', 'danger');
                return;
            }
            if (isNaN(rechargeAmount) || rechargeAmount < 0.01 || rechargeAmount > 1000000) {
                displayMessage(messageDiv, 'Please enter a valid recharge amount between 0.01 and 1,000,000 RWF.', 'danger');
                return;
            }
            if (!csrfToken) {
                displayMessage(messageDiv, 'CSRF token is missing. Please refresh the page and try again.', 'danger');
                return;
            }

            messageDiv.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';

            try {
                const payload = { recharge_amount: rechargeAmount };
                const isClient = document.getElementById('serial_number').hasAttribute('readonly');
                if (!isClient) payload.meter_serial_number = serialNumber;

                const response = await fetch('/api/recharge', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-Token': csrfToken
                    },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();
                if (response.ok) {
                    displayMessage(messageDiv, `${data.message} New balance: RWF ${data.new_balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}`, 'success');
                    rechargeForm.reset();
                } else {
                    const errorMsg = data.details ? `${data.error}: ${data.details.map(e => e.msg || e.message).join(', ')}` : data.error || 'An error occurred during recharge.';
                    displayMessage(messageDiv, errorMsg, 'danger');
                }
            } catch (error) {
                console.error('Recharge failed:', error);
                displayMessage(messageDiv, 'Failed to recharge meter. Please check your connection and try again.', 'danger');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Recharge';
            }
        });
    }

    // --- Prediction Form Logic ---
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const serialNumber = document.getElementById('predictSerialNumber').value.trim();
            const messageDiv = document.getElementById('predictionResult');
            const submitBtn = document.getElementById('submitBtn');
            const csrfToken = getCsrfToken(predictionForm);

            if (!serialNumber) {
                displayMessage(messageDiv, 'Please enter a valid meter serial number.', 'danger');
                return;
            }
            if (!csrfToken) {
                displayMessage(messageDiv, 'CSRF token is missing. Please refresh the page and try again.', 'danger');
                console.error('CSRF token missing for prediction form submission.');
                return;
            }

            messageDiv.classList.add('hidden');
            submitBtn.disabled = true;
            submitBtn.textContent = 'Processing...';

            try {
                const response = await fetch('/predictx', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRF-Token': csrfToken
                    },
                    body: JSON.stringify({ serial_number: serialNumber })
                });

                const data = await response.json();
                if (response.ok) {
                    displayMessage(messageDiv, 
                        data.predicted_payment !== undefined 
                            ? `Predicted next payment: ${data.predicted_payment.toLocaleString('en-US', { minimumFractionDigits: 2 })} RWF`
                            : data.message || 'No prediction available.', 
                        'success'
                    );
                } else {
                    const errorMsg = data.details ? `${data.error}: ${data.details.map(e => e.msg || e.message).join(', ')}` : data.error || 'Error getting prediction.';
                    displayMessage(messageDiv, errorMsg, 'danger');
                }
            } catch (error) {
                console.error('Prediction failed:', error);
                displayMessage(messageDiv, 'Failed to get prediction. Please check your connection and try again.', 'danger');
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Predict';
            }
        });
    }

    // --- Fetch Registered Meters Function ---
    async function fetchRegisteredMeters() {
        const metersTableBody = document.getElementById('metersTableBody');
        const messageDiv = document.getElementById('message');
        if (!metersTableBody) return;

        try {
            const response = await fetch('/api/registered_meters', {
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();

            if (response.ok) {
                metersTableBody.innerHTML = '';
                if (data.length > 0) {
                    data.forEach(meter => {
                        const row = metersTableBody.insertRow();
                        row.insertCell(0).textContent = meter[0];
                        row.insertCell(1).textContent = meter[1];
                        row.insertCell(2).textContent = meter[2];
                        row.insertCell(3).textContent = meter[3].toLocaleString('en-US', { minimumFractionDigits: 2 });
                        row.insertCell(4).textContent = new Date(meter[4]).toLocaleString();

                        const actionsCell = row.insertCell(5);
                        actionsCell.className = 'actions';
                        const historyButton = document.createElement('button');
                        historyButton.textContent = 'View History';
                        historyButton.className = 'bg-blue-500 text-white font-semibold py-1 px-2 rounded hover:bg-blue-600 mr-2';
                        historyButton.onclick = () => window.location.href = `/history?serial_number=${meter[0]}`;
                        actionsCell.appendChild(historyButton);

                        const deleteButton = document.createElement('button');
                        deleteButton.textContent = 'Delete';
                        deleteButton.className = 'bg-red-500 text-white font-semibold py-1 px-2 rounded hover:bg-red-600';
                        deleteButton.onclick = () => deleteMeter(meter[0]);
                        actionsCell.appendChild(deleteButton);
                    });
                } else {
                    displayMessage(messageDiv, 'No registered meters found.', 'info');
                }
            } else {
                const errorMsg = data.details ? `${data.error}: ${data.details.map(e => e.msg || e.message).join(', ')}` : data.error || 'Error fetching registered meters.';
                displayMessage(messageDiv, errorMsg, 'danger');
            }
        } catch (error) {
            console.error('Error fetching meters:', error);
            displayMessage(messageDiv, 'Failed to fetch meters. Please check your connection.', 'danger');
        }
    }

    // --- Fetch Recharge History Function ---
    async function fetchRechargeHistory(serialNumber) {
        const historyTableBody = document.getElementById('historyTableBody');
        const messageDiv = document.getElementById('message');
        if (!historyTableBody) return;

        try {
            const response = await fetch(`/api/recharges?serial_number=${serialNumber}`, {
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();

            if (response.ok) {
                historyTableBody.innerHTML = '';
                if (data.length > 0) {
                    data.forEach(recharge => {
                        const row = historyTableBody.insertRow();
                        row.insertCell(0).textContent = recharge[0].toLocaleString('en-US', { minimumFractionDigits: 2 });
                        row.insertCell(1).textContent = new Date(recharge[1]).toLocaleString();
                    });
                } else {
                    displayMessage(messageDiv, 'No recharge history found for this meter.', 'info');
                }
            } else {
                const errorMsg = data.details ? `${data.error}: ${data.details.map(e => e.msg || e.message).join(', ')}` : data.error || 'Error fetching recharge history.';
                displayMessage(messageDiv, errorMsg, 'danger');
            }
        } catch (error) {
            console.error('Error fetching recharge history:', error);
            displayMessage(messageDiv, 'Failed to fetch recharge history. Please check your connection.', 'danger');
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
                headers: { 'Content-Type': 'application/json' }
            });
            const data = await response.json();

            if (response.ok) {
                displayMessage(messageDiv, data.message, 'success');
                fetchRegisteredMeters();
            } else {
                const errorMsg = data.details ? `${data.error}: ${data.details.map(e => e.msg || e.message).join(', ')}` : data.error || 'Error deleting meter.';
                displayMessage(messageDiv, errorMsg, 'danger');
            }
        } catch (error) {
            console.error('Error deleting meter:', error);
            displayMessage(messageDiv, 'Failed to delete meter. Please check your connection.', 'danger');
        }
    }

    // Initialize page-specific logic
    if (document.getElementById('metersTableBody')) {
        fetchRegisteredMeters();
    }
    if (document.getElementById('historyTableBody')) {
        const urlParams = new URLSearchParams(window.location.search);
        const serialNumber = urlParams.get('serial_number') || (document.getElementById('serial_number')?.value || '');
        if (serialNumber) {
            fetchRechargeHistory(serialNumber);
        }
        const serialInput = document.getElementById('serial_number');
        if (serialInput) {
            serialInput.addEventListener('change', (e) => {
                fetchRechargeHistory(e.target.value.trim());
            });
        }
    }
});
async function fetchPredictionHistory(serialNumber) {
    const response = await fetch(`/api/predicted_payments/${serialNumber}`);
    const data = await response.json();
    if (response.ok && data.labels && data.data) {
        new Chart(document.getElementById('predictionChart'), {
            type: 'line',
            data: {
                labels: data.labels.map(date => new Date(date).toLocaleDateString()),
                datasets: [{
                    label: 'Predicted Payments (RWF)',
                    data: data.data,
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, title: { display: true, text: 'Amount (RWF)' } },
                    x: { title: { display: true, text: 'Date' } }
                }
            }
        });
    }
}

if (predictionForm) {
    const serialNumber = document.getElementById('predictSerialNumber').value.trim();
    if (serialNumber) fetchPredictionHistory(serialNumber);
    document.getElementById('predictSerialNumber').addEventListener('change', (e) => {
        fetchPredictionHistory(e.target.value.trim());
    });
}