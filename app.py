from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Imagescan')  # Define the route for Imagescan.html
def imagescan():
    return render_template('Imagescan.html')

if __name__ == '__main__':
    app.run(debug=True)
