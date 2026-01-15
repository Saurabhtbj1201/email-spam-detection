# Email Spam Detection

A web application for detecting whether an email message is spam or safe using a machine learning model.

## Features

- Classifies email messages as "spam" or "safe"
- Interactive web interface with modern UI
- Displays a table of all emails and their classifications

## Project Structure

- `templates/index.html` - Main web interface
- `data/spam.csv` - Sample dataset of emails
- `app.py` - Flask backend

## Live Demo

- **Frontend:** [https://ems-bice-rho.vercel.app/](https://ems-bice-rho.vercel.app/)
- **Backend API:** [https://esd-z761.onrender.com/](https://esd-z761.onrender.com/)

## Setup (Local Development)

1. **Clone the repository**  
   ```
   git clone https://github.com/Saurabhtbj1201/email-spam-detection
   cd email_spam_detection
   ```

2. **Install dependencies**  
   Make sure you have Python 3.x and pip installed.  
   ```
   pip install -r requirements.txt
   ```

3. **Run the Flask app**  
   ```
   python app.py
   ```

4. **Open the web interface**  
   Open `templates/index.html` in your browser, or deploy the frontend as shown below.

## Deployment

- **Frontend:** Deployed on [Vercel](https://vercel.com/)  
  (see `templates/index.html`)

- **Backend:**
  Deployed on [Render](https://render.com/)  
  [https://esd-z761.onrender.com/](https://esd-z761.onrender.com/)

## Usage

- Enter an email message in the text area and click "Check" to classify it.
- The result will display whether the email is spam or safe.
- All emails and their classifications are shown in a table (if implemented in backend).

## API Endpoints

- `POST /predict`  
  Request JSON: `{ "message": "your email text" }`  
  Response JSON: `{ "message": "...", "prediction": "spam" | "safe" }`

- `GET /`  
  Returns a simple status message.

- `GET /emails`  
  (If implemented) Returns all emails and their classifications.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Developer
<div align="center">

### © Made with ❤️ by Saurabh Kumar. All Rights Reserved 2025

<!-- Profile Section with Photo and Follow Button -->
<a href="https://github.com/Saurabhtbj1201">
  <img src="https://github.com/Saurabhtbj1201.png" width="100" style="border-radius: 50%; border: 3px solid #0366d6;" alt="Saurabh Profile"/>
</a>

### [Saurabh Kumar](https://github.com/Saurabhtbj1201)

<a href="https://github.com/Saurabhtbj1201">
  <img src="https://img.shields.io/github/followers/Saurabhtbj1201?label=Follow&style=social" alt="GitHub Follow"/>
</a>

### 🔗 Connect With Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/saurabhtbj1201)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/saurabhtbj1201)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/saurabhtbj1201)
[![Facebook](https://img.shields.io/badge/Facebook-1877F2?style=for-the-badge&logo=facebook&logoColor=white)](https://facebook.com/saurabh.tbj)
[![Portfolio](https://img.shields.io/badge/Portfolio-FF5722?style=for-the-badge&logo=todoist&logoColor=white)](https://gu-saurabh.site)
[![WhatsApp](https://img.shields.io/badge/WhatsApp-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://wa.me/9798024301)

---

<p align="center">

  <strong>Made with ❤️ by Saurabh Kumar</strong>
  <br>
  ⭐ Star this repo if you find it helpful!
</p>

![Repo Views](https://komarev.com/ghpvc/?username=Saurabhtbj1201&style=flat-square&color=red)

</div>

---

<div align="center">

### 💝 If you like this project, please give it a ⭐ and share it with others!

**Happy Coding! 🚀**

</div>
