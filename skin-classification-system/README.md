# Skin Classification System

This project is a web application for skin classification using machine learning. Users can register, log in, upload images of skin conditions, and receive predictions about the classification of the uploaded images.

## Features

- User registration and login
- Dashboard for image uploads
- Image classification using a trained model
- Display of results including the uploaded image, predicted class, and probability

## Project Structure

```
skin-classification-system
├── src
│   ├── app.py
│   ├── auth
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   └── utils.py
│   ├── dashboard
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── model
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── results
│   │   ├── __init__.py
│   │   └── routes.py
│   ├── static
│   │   └── uploads
│   ├── templates
│   │   ├── dashboard.html
│   │   ├── login.html
│   │   ├── register.html
│   │   └── results.html
│   └── db
│       └── connection.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd skin-classification-system
   ```

2. **Install dependencies:**
   Create a virtual environment and activate it, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Database Configuration:**
   Ensure you have a MySQL database set up. Update the database connection details in `src/db/connection.py`.

4. **Run the application:**
   Start the Flask application by running:
   ```
   python src/app.py
   ```

5. **Access the application:**
   Open your web browser and go to `http://127.0.0.1:5000` to access the application.

## Usage

- **Register:** Create a new account to access the application.
- **Login:** Use your credentials to log in.
- **Dashboard:** Upload images of skin conditions for classification.
- **Results:** View the classification results along with the uploaded image.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.