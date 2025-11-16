


https://github.com/user-attachments/assets/0bc1e910-a272-4ae6-bae5-5bc6291f79fa




# 🍽️ RecipePro - AI-Powered Recipe Generator

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Django](https://img.shields.io/badge/Django-5.1.4-green.svg)](https://djangoproject.com)
[![AI Powered](https://img.shields.io/badge/AI-Groq%20API-orange.svg)](https://groq.com)

Transform your cooking experience with AI-powered recipe generation! RecipePro creates personalized recipes from ingredients, voice input, or food images using cutting-edge artificial intelligence.

## ✨ Features

### 🎤 **Voice Input**
- Simply speak your ingredients or cravings
- Advanced speech recognition technology
- Hands-free cooking experience

### 📸 **Image Recognition**  
- Upload food photos to get instant recipes
- AI analyzes dishes and recreates recipes
- Supports multiple image formats

### 🖼️ **AI-Generated Images**
- Every recipe comes with beautiful AI-generated images
- Multiple image sources: Pollinations, Prodia, Unsplash
- Professional food photography quality

### 🔊 **Audio Recipes**
- Text-to-speech for hands-free cooking
- Listen to instructions while cooking
- Natural voice synthesis

### 💾 **Recipe Management**
- Save your favorite recipes locally
- Browse and organize your collection
- Quick access to saved recipes

### 📤 **Easy Sharing**
- Share recipes via social media
- Copy to clipboard functionality
- Multiple export formats

## 🚀 Technology Stack

- **Backend**: Django 5.1.4, Python 3.10+
- **AI Integration**: Groq API (migrated from Google Gemini), LangChain
- **Frontend**: TailwindCSS, JavaScript ES6+
- **Speech**: Web Speech API
- **Image Generation**: Pollinations.ai, Prodia, Unsplash APIs
- **Database**: SQLite (development)

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Rahul4112002/RecipePro.ai.git
   cd RecipePro.ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   PRODIA_API_KEY=your_prodia_api_key_here  # Optional
   UNSPLASH_ACCESS_KEY=your_unsplash_key_here  # Optional
   SECRET_KEY=your_django_secret_key_here
   ```

5. **Database Setup**
   ```bash
   python manage.py migrate
   python manage.py collectstatic
   ```

6. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

7. **Access Application**
   Open your browser and navigate to `http://127.0.0.1:8000`

## 🔑 API Keys Setup

### Required: Groq API Key
1. Visit [Groq Console](https://console.groq.com)
2. Create an account and generate an API key
3. Add to `.env` file: `GROQ_API_KEY=your_key_here`

### Optional: Enhanced Features
- **Prodia API**: For AI image generation
- **Unsplash API**: For stock food photos

## 👨‍💻 Developer

**Rahul Chauhan**
- 🌐 Portfolio: [rahul4112.me](https://rahul4112.me)
- 💻 GitHub: [@Rahul4112002](https://github.com/Rahul4112002)
- 💼 LinkedIn: [Rahul Chauhan](https://linkedin.com/in/rahul-chauhan-932522230)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

---

<div align="center">

**Made with ❤️ by [Rahul Chauhan](https://rahul4112.me)**

**🍽️ Transform your cooking with AI!**

</div>
