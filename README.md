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

## How to Deploy to GitHub

1. **Create a new repository on GitHub**  
   Go to [https://github.com/new](https://github.com/new) and create a new repository (e.g., `email_spam_detection`).

2. **Initialize git in your project folder**  
   If you haven't already, open a terminal in your project directory and run:
   ```
   git init
   ```

3. **Add all project files to git**  
   ```
   git add .
   ```

4. **Commit your changes**  
   ```
   git commit -m "Initial commit"
   ```

5. **Add the GitHub repository as a remote**  
   Replace `<your-username>` and `<repo-name>` with your actual GitHub username and repository name:
   ```
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   ```

6. **Push your code to GitHub**  
   ```
   git push -u origin master
   ```
   If you are using `main` as the default branch, use:
   ```
   git push -u origin main
   ```

7. **Check your repository on GitHub**  
   Visit your repository URL to see your code online.

## Notes

- The backend should provide `/predict` and `/emails` endpoints.
- The dataset can be expanded for better accuracy.

## License

MIT License
