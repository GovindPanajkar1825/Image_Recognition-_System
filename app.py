from flask import Flask, render_template, request, redirect, url_for
import os
from model import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            predictions = predict_image(filepath)
            return render_template("index.html", predictions=predictions, filename=file.filename)
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == "__main__":
    app.run(debug=True)
