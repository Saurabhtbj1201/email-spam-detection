# Email Spam Detection

A web application for detecting whether an email message is spam or safe using a machine learning model.

## Features

- Classifies email messages as "spam" or "safe"
- Interactive web interface with modern UI
- Displays a table of all emails and their classifications

## Project Structure

- `templates/index.html` - Main web interface
- `data/spam.csv` - Sample dataset of emails
- `app.py` (or similar) - Flask backend (not included here)

## Setup

1. **Clone the repository**  
   ```
   git clone <your-repo-url>
   cd email_spam_detection
   ```

2. **Install dependencies**  
   Make sure you have Python 3.x and pip installed.  
   ```
   pip install flask scikit-learn pandas
   ```

3. **Run the Flask app**  
   ```
   python app.py
   ```

4. **Open the web interface**  
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Usage

- Enter an email message in the text area and click "Check" to classify it.
- The result will display whether the email is spam or safe.
- All emails and their classifications are shown in a table (if implemented in backend).