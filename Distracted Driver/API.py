from flask import Flask, request, jsonify
import base64

app = Flask(__name__)
exi
@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image file from the request
    image_file = request.files.get('image')
    
    if image_file is None:
        return jsonify({'error': 'No image provided'})

    # Read the image file and convert it to base64
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Process the image (example: return the image as JSON)
    result = {'image_data': image_data}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
