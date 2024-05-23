from flask import Flask, render_template, jsonify
from cassandra.cluster import Cluster

app = Flask(__name__)

# Connect to Cassandra
cluster = Cluster(['localhost'])  # Change this to your Cassandra cluster address if different
session = cluster.connect('churn_stream')  # Change 'telecom_data' to your keyspace name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    # Query data from Cassandra
    rows = session.execute(
        "SELECT state, area_code, label_churn, total_day_minutes FROM churn_predictions")

    # Convert rows to a list of dictionaries
    data = [{"state": row.state, "area_code": row.area_code, "label_churn": row.label_churn, "total_day_minutes": row.total_day_minutes} for row in rows]

    return jsonify(data)

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)
