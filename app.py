from flask import Flask, render_template, request, redirect, url_for
import os
from skimage.metrics import structural_similarity
import cv2
import imutils

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Check if files are uploaded
        if 'original' not in request.files or 'tampered' not in request.files:
            return "Please upload both original and tampered images!"

        # Retrieve uploaded files
        original_file = request.files['original']
        tampered_file = request.files['tampered']

        # Save files
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], "original.png")
        tampered_path = os.path.join(app.config['UPLOAD_FOLDER'], "tampered.png")
        original_file.save(original_path)
        tampered_file.save(tampered_path)

        # Process images
        original = cv2.imread(original_path)
        tampered = cv2.imread(tampered_path)

        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

        (score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Save output images
        static_dir = "static"
        os.makedirs(static_dir, exist_ok=True)
        original_with_contours_path = os.path.join(static_dir, "original_with_contours.png")
        tampered_with_contours_path = os.path.join(static_dir, "tampered_with_contours.png")
        diff_path = os.path.join(static_dir, "diff.png")
        thresh_path = os.path.join(static_dir, "thresh.png")

        cv2.imwrite(original_with_contours_path, original)
        cv2.imwrite(tampered_with_contours_path, tampered)
        cv2.imwrite(diff_path, diff)
        cv2.imwrite(thresh_path, thresh)

        # Render results
        return render_template(
            'results.html',
            score=score,
            original=url_for('static', filename='original_with_contours.png'),
            tampered=url_for('static', filename='tampered_with_contours.png'),
            diff=url_for('static', filename='diff.png'),
            thresh=url_for('static', filename='thresh.png'),
        )
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
