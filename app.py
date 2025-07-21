from flask import Flask, request, jsonify
import os
import tempfile
from werkzeug.utils import secure_filename
from heart_rate_analysis import get_heart_rate_from_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video = request.files['video']
    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    try:
        result = get_heart_rate_from_video(video_path)
        if result is None:
            return jsonify({'error': 'Heart rate analysis failed'}), 500
        return jsonify({'heart_rate_bpm': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

