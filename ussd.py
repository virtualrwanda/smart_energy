from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/ussd', methods=['POST'])
def ussd_callback():
    session_id = request.form.get("sessionId")
    service_code = request.form.get("serviceCode")
    phone_number = request.form.get("phoneNumber")
    text = request.form.get("text")

    # Split user input into levels
    user_input = text.strip().split("*")

    if text == "":
        response = "CON Welcome to Energy Meter Service\n"
        response += "1. Check Meter Balance\n"
        response += "2. Recharge Meter\n"
        return response

    elif text == "1":
        response = "CON Enter your meter serial number:"
        return response

    elif len(user_input) == 2 and user_input[0] == "1":
        serial_number = user_input[1]
        conn = sqlite3.connect('energy_meter.db')
        cursor = conn.cursor()
        cursor.execute("SELECT balance FROM meter_data WHERE serial_number = ?", (serial_number,))
        result = cursor.fetchone()
        conn.close()

        if result:
            balance = result[0]
            response = f"END Your meter balance is: {balance} RWF"
        else:
            response = "END Meter not found. Please check your serial number."
        return response

    elif text == "2":
        response = "CON Enter your meter serial number:"
        return response

    elif len(user_input) == 2 and user_input[0] == "2":
        response = "CON Enter recharge amount:"
        return response

    elif len(user_input) == 3 and user_input[0] == "2":
        serial_number = user_input[1]
        try:
            amount = float(user_input[2])
        except ValueError:
            return "END Invalid recharge amount."

        conn = sqlite3.connect('energy_meter.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO recharges (serial_number, recharge_amount, timestamp) VALUES (?, ?, datetime('now'))",
                       (serial_number, amount))
        cursor.execute("UPDATE meter_data SET balance = balance + ? WHERE serial_number = ?", (amount, serial_number))
        conn.commit()
        conn.close()

        response = f"END Recharge of {amount} RWF successful for meter {serial_number}."
        return response

    else:
        response = "END Invalid input. Please try again."
        return response

if __name__ == '__main__':
    app.run(port=5000, debug=True)
