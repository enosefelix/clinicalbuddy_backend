import os
import io
import requests
import hashlib
from helpers.constants import LOCAL_FRONT_END_URL, PRODUCTION_FRONT_END_URL
from werkzeug.utils import secure_filename
from helpers.helpers import conversation_chain
from datetime import timedelta
from flask import Flask, jsonify, request, send_file
from openai import OpenAI
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    verify_jwt_in_request,
)


# Load environment variables
load_dotenv()
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")

FRONT_END_URLS = [LOCAL_FRONT_END_URL, PRODUCTION_FRONT_END_URL]


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": FRONT_END_URLS}})


bcrypt = Bcrypt()
openAIClient = OpenAI()
secret_key = os.environ.get("JWT_SECRET_KEY")
app.config["JWT_SECRET_KEY"] = secret_key
jwt = JWTManager(app)


from qdrant.qdrant import (
    qdrant_vector_embedding,
    delete_selected_pdf_from_qdrant,
    delete_all_pdfs_by_user_from_qdrant,
    delete_all_selected_pdf_array_from_qdrant,
)

from firestore.firestore import (
    fetch_missing_pdfs_from_firestore,
    upload_missing_pdfs_to_firestore,
    upload_user_to_firestore,
    fetch_all_users_from_firestore,
    fetch_bookmarks_from_firestore,
    delete_pdf_from_missing_pdfs,
    delete_bookmark_from_firestore,
    delete_user_from_firestore,
    update_missing_pdfs_category_to_firestore,
    delete_multiple_pdfs_from_firestore,
    fetch_user_by_username,
    add_bookmark_to_firestore,
    update_accept_disclaimer_field,
)

from helpers.helpers import upload_pdf_to_qdrant, upload_pdf_to_s3bucket_and_get_info


system_prompt = "You are a helpful assistant for an AI assisted chat bot that helps users search through clinical and medical guidelines"


def transcribe_audio(file_bytes, file_type, content_type):
    file_buffer = io.BytesIO(file_bytes)
    file_info = ("temp." + file_type, file_buffer, content_type)
    transcript = openAIClient.audio.transcriptions.create(
        model="whisper-1",
        file=file_info,
        response_format="text",
    )

    return transcript

    # corrected_transcript = openAIClient.chat.completions.create(
    #     model="gpt-3.5-turbo-0125",
    #     temperature=0,
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": transcript},
    #     ],
    # )

    # return corrected_transcript.choices[0].message.content


@app.route("/api/upload-audio", methods=["POST", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def upload_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No file part"}), 400

    audio_file = request.files["audio"]

    if audio_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if audio_file:
        filename = secure_filename(audio_file.filename)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        audio_file.save(audio_path)
        with open(audio_path, "rb") as file:
            file_bytes = file.read()

        file_type = "mp3"
        content_type = "audio/mp3"
        transcription = transcribe_audio(file_bytes, file_type, content_type)
        os.remove(audio_path)

        return jsonify({"transcription": transcription}), 200


@jwt_required()
@app.route("/api/conversation_chain", methods=["POST", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def get_conversation_chain():
    verify_jwt_in_request()
    data = request.get_json()
    user_question = data.get("question")
    selected_pdf = data.get("selected_pdf")
    cluster = data.get("cluster")
    user_name = get_jwt_identity()

    return conversation_chain(
        user_question,
        selected_pdf,
        qdrant_vector_embedding,
        cluster,
        user_name,
    )


@jwt_required()
@app.route("/api/bookmarks", methods=["GET", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def get_bookmarks():
    verify_jwt_in_request()
    user_name = get_jwt_identity()
    response = fetch_bookmarks_from_firestore(user_name)
    if len(response) > 0:
        return jsonify(
            {
                "bookmarks": response,
                "status": 200,
            }
        )
    else:
        return jsonify(
            {
                "bookmarks": [],
                "status": 204,
            }
        )


@jwt_required()
@app.route("/api/delete_bookmark", methods=["DELETE", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def delete_bookmark():
    verify_jwt_in_request()
    data = request.get_json()
    user_name = get_jwt_identity()
    bookmark_id = data.get("bookmark_id")

    if bookmark_id:
        response = delete_bookmark_from_firestore(bookmark_id, user_name)
        if response:
            return jsonify(
                {
                    "deleted_bookmark": delete_bookmark_from_firestore(
                        bookmark_id, user_name
                    ),
                    "status": 200,
                }
            )
    else:
        return jsonify({"status": 400})


@jwt_required()
@app.route("/api/add_bookmark", methods=["POST", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def add_bookmark():
    verify_jwt_in_request()
    bookmark = request.get_json()

    if bookmark:
        response = add_bookmark_to_firestore(bookmark)

        if response:
            return jsonify(
                {
                    # "bookmark": response,
                    "status": 200,
                }
            )
    else:
        return jsonify({"status": 400})


# Pdfs
@app.route("/api/get_pdf", methods=["GET", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def get_pdf():
    try:
        # Get the URL parameter from the request
        pdf_url = request.args.get("url")

        # Fetch the PDF content from the specified URL
        pdf_response = requests.get(pdf_url)
        pdf_content = pdf_response.content

        # Set the Content-Disposition header for inline display
        response = send_file(
            io.BytesIO(pdf_content),
            as_attachment=False,  # Set to True if you want to force download
            download_name="example.pdf",
            mimetype="application/pdf",
        )

        return response

    except Exception as e:
        return jsonify({"status": 400, "error": str(e)})


@jwt_required()
@app.route("/api/missing_pdfs", methods=["GET", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def get_missing_pdfs():
    verify_jwt_in_request()
    cluster = request.args.get("cluster", type=str)
    user_name = get_jwt_identity()

    response = fetch_missing_pdfs_from_firestore(cluster, user_name)
    if len(response) > 0:
        return jsonify(
            {
                "pdfs": fetch_missing_pdfs_from_firestore(cluster, user_name),
                "status": 200,
            }
        )
    else:
        return jsonify(
            {
                "pdfs": [],
                "status": 204,
            }
        )


@jwt_required()
@app.route("/api/update_multiple_missing_pdfs_category", methods=["PATCH", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def update_multiple_missing_pdfs_category():
    verify_jwt_in_request()
    data = request.get_json()
    selected_pdfs = data.get("selected_pdfs")
    selected_category = data.get("selected_category")

    if selected_pdfs != "" and len(selected_category) > 0:
        update_missing_pdfs_category_to_firestore(selected_pdfs, selected_category)

        return jsonify(
            {
                "status": 200,
            }
        )
    else:
        return jsonify({"status": 400})


@jwt_required()
@app.route("/api/delete_missing_pdf", methods=["DELETE", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def delete_missing_pdf():
    verify_jwt_in_request()
    data = request.get_json()
    pdf_name = data.get("pdf_name")
    user_name = get_jwt_identity()

    if pdf_name:
        delete_selected_pdf_from_qdrant(
            pdf_name,
            user_name,
        )
        delete_pdf_from_missing_pdfs(pdf_name, user_name)

        return jsonify(
            {
                "deleted_pdf": delete_pdf_from_missing_pdfs(pdf_name, user_name),
                "status": 200,
            }
        )
    else:
        return jsonify({"status": 400})


@jwt_required()
@app.route("/api/upload_pdfs", methods=["POST", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def upload_pdf():
    try:
        verify_jwt_in_request()
        user_name = get_jwt_identity()
        new_pdfs = []

        existing_pdfs = []
        new_pdf_file_objs = []
        if "files" not in request.files:
            return jsonify({"status": 400, "message": "No files provided"})
        uploaded_files = request.files.getlist("files")
        category = request.form.get("category")
        cluster = request.form.get("cluster")
        fetched_pdfs = fetch_missing_pdfs_from_firestore(cluster, user_name)

        for uploaded_file in uploaded_files:
            pdf_name = uploaded_file.filename
            pdf_content = uploaded_file.read()
            pdf_hash = hashlib.sha256(pdf_content).hexdigest()
            pdf_exists = any(
                pdf["pdf_name"] == pdf_name or pdf["hash_value"] == pdf_hash
                for pdf in fetched_pdfs
            )

            if pdf_exists:
                existing_pdfs.append(pdf_name)
            else:
                new_pdf_file_objs.append(uploaded_file)

        upload_pdf_to_qdrant(new_pdf_file_objs, cluster, category, user_name)
        for uploaded_file in new_pdf_file_objs:
            pdf_name = uploaded_file.filename
            pdf_info = upload_pdf_to_s3bucket_and_get_info(
                uploaded_file, cluster, user_name, category
            )
            new_pdfs.append(pdf_info)

        upload_missing_pdfs_to_firestore(new_pdfs)

        return jsonify(
            {
                "status": 200,
                "existing_pdfs": existing_pdfs,
                "new_pdfs": new_pdfs,
            }
        )

    except Exception as e:
        return jsonify({"status": 500, "message": "Internal Server Error"})


@jwt_required()
@app.route("/api/delete_multiple_pdfs", methods=["DELETE", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def delete_multiple_pdfs():
    verify_jwt_in_request()
    user_name = get_jwt_identity()
    data = request.get_json()
    selected_pdfs = data.get("selected_pdfs")
    response = delete_multiple_pdfs_from_firestore(selected_pdfs)
    delete_all_selected_pdf_array_from_qdrant(selected_pdfs, user_name)

    if response:
        return response
    else:
        return jsonify({"status": 400})


@app.route("/api/register", methods=["POST", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def register():
    data = request.get_json()

    user_name = data.get("username")
    password = data.get("password")
    name = data.get("fullname")
    cluster = data.get("cluster")
    role = data.get("role")
    public_cluster = data.get("publicCluster")

    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
    formatted_cluster = (
        f"{user_name}_cluster" if cluster == "personal_cluster" else public_cluster
    )

    new_user = {
        "name": name,
        "user_name": user_name,
        "cluster": formatted_cluster,
        "role": role,
        "password": hashed_password,
        "accept_disclaimer": False,
    }

    return upload_user_to_firestore(new_user)


@app.route("/api/login", methods=["POST", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def login():
    data = request.get_json()
    user_name = data.get("username")
    password = data.get("password")
    accept_disclaimer = data.get("accept_disclaimer")
    user_obj = fetch_user_by_username(user_name)
    all_users = fetch_all_users_from_firestore()

    for user_data in all_users:
        if user_data["user_name"] == user_name and bcrypt.check_password_hash(
            user_data["password"], password
        ):
            # User authenticated, generate JWT
            access_token = create_access_token(
                identity=user_name,
                expires_delta=timedelta(minutes=120),
                additional_claims={
                    "user": {
                        "cluster": user_obj.get("cluster"),
                        "role": user_obj.get("role"),
                        "user_name": user_obj.get("user_name"),
                        "name": user_obj.get("name"),
                        "accept_disclaimer": user_obj.get("accept_disclaimer"),
                    }
                },
            )
            refresh_token = create_refresh_token(identity=user_name)
            update_accept_disclaimer_field(user_name, accept_disclaimer)
            return jsonify(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "status": 200,
                }
            )

    # User not found or authentication failed
    return jsonify({"status": 401})


@jwt_required()
@app.route("/api/users", methods=["GET", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def get_users():
    verify_jwt_in_request()
    user_name = get_jwt_identity()
    if user_name == SUPER_ADMIN_USERNAME:
        response = fetch_all_users_from_firestore()
        if len(response) > 0:
            return jsonify({"users": response, "status": 200})
        else:
            return jsonify({"users": [], "status": 400})

    else:
        return jsonify(
            {
                "status": 401,
            }
        )


@jwt_required()
@app.route("/api/delete_user", methods=["DELETE", "OPTIONS"])
@cross_origin(origins=FRONT_END_URLS, headers=["Content-Type", "Authorization"])
def delete_user():
    verify_jwt_in_request()
    current_user = get_jwt_identity()
    data = request.get_json()
    user_name = data.get("user_name")

    if current_user == SUPER_ADMIN_USERNAME:
        delete_all_pdfs_by_user_from_qdrant(user_name)
        response = delete_user_from_firestore(user_name)
        if response:
            return response
        else:
            return jsonify({"status": 400})
    else:
        return jsonify(
            {
                "status": 401,
            }
        )


if __name__ == "__main__":
    app.config["UPLOAD_FOLDER"] = "uploads"
    # os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
