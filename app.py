import os
import logging
from config.constants import LOCAL_FRONT_END_URL, PRODUCTION_FRONT_END_URL
from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager,
)

# Load environment variables
load_dotenv()
FRONT_END_URLS = [LOCAL_FRONT_END_URL, PRODUCTION_FRONT_END_URL]


# App config
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": FRONT_END_URLS}})
app.config["CORS_HEADERS"] = "Content-Type, Authorization"
logging.basicConfig(filename="clinicalbuddy.log", level=logging.DEBUG)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
secret_key = os.environ.get("JWT_SECRET_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app.config["JWT_SECRET_KEY"] = secret_key
jwt = JWTManager(app)


from routes.routes import Routes

routes = Routes(app)
routes.register_routes()

if __name__ == "__main__":
    app.run(debug=True)
