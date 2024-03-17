import os
from dotenv import load_dotenv
from google.cloud import firestore, storage
from flask import jsonify, make_response
from functools import lru_cache


# Load environment variables and initialise firestore
load_dotenv()


# Environment Variables
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME")
FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_KEY_PATH"
)


# Define constants
FIRESTORE_COLLECTION_NAME = "admin_collection"
FIRESTORE_DOCUMENT_NAME = "admin_collection_id"
FIRESTORE_REFERENCE_NAME = "missing_pdfs"
FIRESTORE_USER_COLLECTION_NAME = "user_collection"
FIRESTORE_USER_DOCUMENT_NAME = "user_collection_id"
FIRESTORE_USER_REFERENCE_NAME = "users"
FIRESTORE_BOOKMARK_COLLECTION_NAME = "bookmark_collection"
FIRESTORE_BOOKMARK_DOCUMENT_NAME = "bookmark_collection_id"
FIRESTORE_BOOKMARK_REFERENCE_NAME = "bookmarks"
FIRESTORE_FLOWCHART_COLLECTION_NAME = "flowchart_collection"
FIRESTORE_FLOWCHART_DOCUMENT_NAME = "flowchart_collection_id"
FIRESTORE_FLOWCHART_REFERENCE_NAME = "flowcharts"


# Firebase Config
db = firestore.Client()


def initialize_storage_client():
    return storage.Client()


storage_client = initialize_storage_client()
bucket = storage_client.bucket(FIREBASE_STORAGE_BUCKET)


# # Flowcharts
# def upload_flowcharts_to_firestore(flow_charts):
#     try:
#         doc_ref = db.collection(FIRESTORE_FLOWCHART_COLLECTION_NAME).document(
#             FIRESTORE_FLOWCHART_DOCUMENT_NAME
#         )

#         # Get the current data from Firestore
#         doc = doc_ref.get()

#         # Check if the document exists
#         if doc.exists:
#             doc_data = doc.to_dict()

#             if doc_data:
#                 # If the document exists, update the "flow_charts" field with new data
#                 existing_flow_charts = doc_data.get(
#                     FIRESTORE_FLOWCHART_REFERENCE_NAME, []
#                 )

#                 # Check for duplicates based on the "url" field
#                 existing_urls = {
#                     flow_chart["url"] for flow_chart in existing_flow_charts
#                 }
#                 new_flow_charts = [
#                     flow_chart
#                     for flow_chart in flow_charts
#                     if flow_chart["url"] not in existing_urls
#                 ]
#                 duplicate_urls = [
#                     flow_chart["url"]
#                     for flow_chart in flow_charts
#                     if flow_chart["url"] in existing_urls
#                 ]

#                 if duplicate_urls:
#                     st.warning(
#                         f"Flowcharts with these URLs already exist in the database: {', '.join(duplicate_urls)}"
#                     )
#                     return "error"

#                 # Update the "flow_charts" field with new and unique data
#                 updated_flow_charts = existing_flow_charts + new_flow_charts
#                 doc_data[FIRESTORE_FLOWCHART_REFERENCE_NAME] = updated_flow_charts

#                 # Update the Firestore document with the merged data
#                 doc_ref.set(doc_data)
#                 st.success(f"{len(new_flow_charts)} files successfully uploaded")
#                 return new_flow_charts
#             else:
#                 st.warning("Document data is empty.")
#                 return "error"

#         else:
#             # If the document doesn't exist, create a new one with the new data
#             doc_ref.set({FIRESTORE_FLOWCHART_REFERENCE_NAME: flow_charts})
#             return flow_charts

#     except Exception as upload_error:
#         st.error(
#             f"Error uploading missing flowcharts to Firestore: {str(upload_error)}"
#         )
#         return "error"


# def fetch_flowcharts_from_firestore(user_name):
#     try:
#         # Reference to the Firestore document
#         doc_ref = db.collection(FIRESTORE_FLOWCHART_COLLECTION_NAME).document(
#             FIRESTORE_FLOWCHART_DOCUMENT_NAME
#         )

#         # Get the document data
#         doc = doc_ref.get()

#         # Check if the document exists
#         if doc.exists:
#             doc_data = doc.to_dict()

#             if doc_data and FIRESTORE_FLOWCHART_REFERENCE_NAME in doc_data:
#                 flowcharts = doc_data[FIRESTORE_FLOWCHART_REFERENCE_NAME]

#                 # Filter flowcharts based on the user_name
#                 filtered_flowcharts = [
#                     item for item in flowcharts if item.get("user_name") == user_name
#                 ]

#                 if filtered_flowcharts:
#                     return filtered_flowcharts
#                 else:
#                     return []

#             else:
#                 st.error("Document data is empty.")

#         else:
#             st.error("Document does not exist.")

#     except Exception as fetch_error:
#         st.error(f"Error fetching flowcharts from Firestore: {str(fetch_error)}")
#         return []


# Users


def fetch_user_by_username(user_name):
    try:
        # Reference to the Firestore document
        doc_ref = db.collection(FIRESTORE_USER_COLLECTION_NAME).document(
            FIRESTORE_USER_DOCUMENT_NAME
        )

        # Get the document data
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data and FIRESTORE_USER_REFERENCE_NAME in doc_data:
                all_users = doc_data[FIRESTORE_USER_REFERENCE_NAME]

                # Find the user with the specified username
                user_object = next(
                    (user for user in all_users if user.get("user_name") == user_name),
                    None,
                )

                if user_object:
                    return user_object
                else:
                    return make_response(jsonify({"message": "User not found"}), 404)
            else:
                return make_response(jsonify({"message": "No users found"}), 404)
        else:
            return make_response(
                jsonify({"message": "Firestore document not found"}), 404
            )

    except Exception as fetch_error:
        return jsonify({"status": 400, "error": str(fetch_error)})


def upload_user_to_firestore(new_user):
    try:
        doc_ref = db.collection(FIRESTORE_USER_COLLECTION_NAME).document(
            FIRESTORE_USER_DOCUMENT_NAME
        )

        # Get the current data from Firestore
        doc = doc_ref.get()

        # Extract existing users from the document or create an empty list
        existing_users = doc.to_dict().get(FIRESTORE_USER_REFERENCE_NAME, [])

        # Check if the user already exists
        user_exists = any(
            existing_user["user_name"] == new_user["user_name"]
            for existing_user in existing_users
        )

        if not user_exists:
            # If the user doesn't exist, add the new user
            updated_users = existing_users + [new_user]
            # Update the Firestore document with the merged data
            doc_ref.set({FIRESTORE_USER_REFERENCE_NAME: updated_users})
            return make_response(
                jsonify({"message": "User successfully registered", "status": 200}),
            )

        else:
            return make_response(
                jsonify({"message": "User already exists", "status": 409})
            )

    except firestore.exceptions.NotFound:
        return make_response(
            jsonify({"message": "Firestore document not found", "status": 404})
        )

    except firestore.exceptions.InvalidArgument:
        return make_response(
            jsonify(
                {"message": "Invalid argument for Firestore operation", "status": 400}
            )
        )

    except Exception as upload_error:
        return jsonify({"status": 400})


def fetch_all_users_from_firestore():
    try:
        # Reference to the Firestore document
        doc_ref = db.collection(FIRESTORE_USER_COLLECTION_NAME).document(
            FIRESTORE_USER_DOCUMENT_NAME
        )

        # Get the document data
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data and FIRESTORE_USER_REFERENCE_NAME in doc_data:
                all_users = doc_data[FIRESTORE_USER_REFERENCE_NAME]

                return all_users
            else:
                return make_response(jsonify({"message": "No users found"}), 404)
        else:
            return make_response(
                jsonify({"message": "Firestore document not found"}), 404
            )

    except Exception as fetch_error:
        return jsonify({"status": 400})


def delete_user_from_firestore(user_name):
    try:
        # Retrieve the document from Firestore
        doc_ref = db.collection(FIRESTORE_USER_COLLECTION_NAME).document(
            FIRESTORE_USER_DOCUMENT_NAME
        )
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                existing_users = doc_data.get(FIRESTORE_USER_REFERENCE_NAME, [])

                # Filter the existing users array to exclude the matching user_name
                updated_users = [
                    user for user in existing_users if user["user_name"] != user_name
                ]

                # Update the Firestore document with the filtered data
                doc_data[FIRESTORE_USER_REFERENCE_NAME] = updated_users
                doc_ref.set(doc_data)
                return jsonify({"status": 200})

            else:
                return jsonify({"status": 400})

        else:
            return jsonify({"status": 400})

    except Exception as delete_error:
        return jsonify({"status": 400})


def update_accept_disclaimer_field(user_name, accept_disclaimer):

    try:
        doc_ref = db.collection(FIRESTORE_USER_COLLECTION_NAME).document(
            FIRESTORE_USER_DOCUMENT_NAME
        )

        # Get the current data from Firestore
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                # If the document exists, update the "missing_pdfs" field with new data
                existing_users = doc_data.get(FIRESTORE_USER_REFERENCE_NAME, [])

                for user in existing_users:
                    if user.get("user_name") == user_name:
                        # Check if accept_disclaimer is already True
                        if not user.get("accept_disclaimer"):
                            # Update the accept_disclaimer field
                            user["accept_disclaimer"] = accept_disclaimer

                            # Update the Firestore document with the modified data
                            doc_ref.set({FIRESTORE_USER_REFERENCE_NAME: existing_users})

                            return make_response(
                                jsonify(
                                    {
                                        "message": f"Accept Disclaimer field updated for {user_name}",
                                        "status": 200,
                                    }
                                ),
                            )
                        else:
                            return make_response(
                                jsonify(
                                    {
                                        "message": f"Accept Disclaimer is already True for {user_name}",
                                        "status": 200,
                                    }
                                ),
                            )

                return make_response(
                    jsonify(
                        {
                            "message": f"User with username {user_name} not found",
                            "status": 404,
                        }
                    )
                )

        return make_response(
            jsonify({"message": "Firestore document not found", "status": 404})
        )

    except Exception as e:
        return make_response(
            jsonify(
                {
                    "message": f"Error updating Accept Disclaimer field: {str(e)}",
                    "status": 500,
                }
            )
        )


# Bookmarks
def fetch_bookmarks_from_firestore(user_name):
    try:
        # Reference to the Firestore document
        doc_ref = db.collection(FIRESTORE_BOOKMARK_COLLECTION_NAME).document(
            FIRESTORE_BOOKMARK_DOCUMENT_NAME
        )

        # Get the document data
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data and FIRESTORE_BOOKMARK_REFERENCE_NAME in doc_data:
                bookmarks = doc_data[FIRESTORE_BOOKMARK_REFERENCE_NAME]

                # Filter bookmarks based on the user_name
                filtered_bookmarks = [
                    item for item in bookmarks if item.get("user_name") == user_name
                ]
                if filtered_bookmarks:
                    return filtered_bookmarks
                else:
                    return []

            else:
                return []

        else:
            return []

    except Exception as fetch_error:
        return []


def delete_bookmark_from_firestore(bookmark_id, user_name):
    try:
        # Retrieve the document from Firestore
        doc_ref = db.collection(FIRESTORE_BOOKMARK_COLLECTION_NAME).document(
            FIRESTORE_BOOKMARK_DOCUMENT_NAME
        )
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                existing_bookmarks = doc_data.get(FIRESTORE_BOOKMARK_REFERENCE_NAME, [])

                # Filter the bookmarks array to exclude the matching bookmark_id and user_name
                updated_bookmarks = [
                    item
                    for item in existing_bookmarks
                    if item.get("bookmark_id") != bookmark_id
                    or (item.get("user_name") != user_name)
                ]

                # Update the Firestore document with the filtered data
                doc_data[FIRESTORE_BOOKMARK_REFERENCE_NAME] = updated_bookmarks
                doc_ref.set(doc_data)

                return bookmark_id

            else:
                return jsonify({"status": 400})

        else:
            return jsonify({"status": 400})

    except Exception as delete_error:
        return jsonify({"status": 400})


def add_bookmark_to_firestore(bookmark):
    try:
        doc_ref = db.collection(FIRESTORE_BOOKMARK_COLLECTION_NAME).document(
            FIRESTORE_BOOKMARK_DOCUMENT_NAME
        )

        # Get the current data from Firestore
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                # If the document exists, update the "BOOKMARKS ARR" field with new data
                existing_bookmarks = doc_data.get(FIRESTORE_BOOKMARK_REFERENCE_NAME, [])
                updated_bookmarks = existing_bookmarks + [bookmark]
                doc_data[FIRESTORE_BOOKMARK_REFERENCE_NAME] = updated_bookmarks

                # Update the Firestore document with the merged data
                doc_ref.set(doc_data)
                return jsonify({"bookmark": bookmark, "status": 200})
            else:
                return jsonify({"status": 400, "message": "Document data is empty"})

        else:
            # If the document doesn't exist, create a new one with the new data
            doc_ref.set({FIRESTORE_BOOKMARK_REFERENCE_NAME: [bookmark]})
            return jsonify({"bookmark": bookmark, "status": 200})

    except Exception as upload_error:
        return jsonify({"status": 400, "message": str(upload_error)})


def fetch_missing_pdfs_from_firestore(cluster, user_name):
    try:
        # Reference to the Firestore document
        doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).document(
            FIRESTORE_DOCUMENT_NAME
        )

        # Get the document data
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data and FIRESTORE_REFERENCE_NAME in doc_data:
                missing_pdfs = doc_data[FIRESTORE_REFERENCE_NAME]

                # Filter missing_pdfs based on the cluster
                filtered_missing_pdfs = [
                    item for item in missing_pdfs if item.get("cluster") == cluster
                ]

                # Return all missing_pdfs if the current user is a super admin
                if cluster == "admin_cluster":
                    return missing_pdfs
                else:
                    return filtered_missing_pdfs

        else:
            return jsonify({"status": 400})

    except Exception as fetch_error:
        return jsonify({"status": 400})


cached_missing_pdfs = None


# Function to fetch and cache missing PDFs from Firestore
@lru_cache(maxsize=128)
def fetch_cached_missing_pdfs(cluster, user_name):
    global cached_missing_pdfs
    if cached_missing_pdfs is None:
        cached_missing_pdfs = fetch_missing_pdfs_from_firestore(cluster, user_name)
    else:
        print("missing pdfs>>>", cached_missing_pdfs)

    return cached_missing_pdfs


def upload_missing_pdfs_to_firestore(missing_pdfs):
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).document(
            FIRESTORE_DOCUMENT_NAME
        )

        # Get the current data from Firestore
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                # If the document exists, update the "missing_pdfs" field with new data
                existing_missing_pdfs = doc_data.get(FIRESTORE_REFERENCE_NAME, [])
                updated_missing_pdfs = existing_missing_pdfs + missing_pdfs
                doc_data[FIRESTORE_REFERENCE_NAME] = updated_missing_pdfs

                # Update the Firestore document with the merged data
                doc_ref.set(doc_data)
                return missing_pdfs
            else:
                return []

        else:
            # If the document doesn't exist, create a new one with the new data
            doc_ref.set({FIRESTORE_REFERENCE_NAME: missing_pdfs})
            return missing_pdfs

    except Exception as upload_error:
        return []


def delete_pdf_from_missing_pdfs(pdf_name, user_name):
    try:
        # Retrieve the document from Firestore
        doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).document(
            FIRESTORE_DOCUMENT_NAME
        )
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                existing_missing_pdfs = doc_data.get(FIRESTORE_REFERENCE_NAME, [])

                # Filter the missing_pdfs array to exclude the matching pdf_name
                updated_missing_pdfs = [
                    item
                    for item in existing_missing_pdfs
                    if item.get("pdf_name") != pdf_name
                    or (item.get("user_name") != user_name)
                ]

                # Update the Firestore document with the filtered data
                doc_data[FIRESTORE_REFERENCE_NAME] = updated_missing_pdfs
                doc_ref.set(doc_data)

                return pdf_name

            else:
                return jsonify({"status": 400})

        else:
            return jsonify({"status": 400})

    except Exception as delete_error:
        return jsonify({"status": 400})


def update_missing_pdfs_category_to_firestore(selected_pdfs, selected_category):
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).document(
            FIRESTORE_DOCUMENT_NAME
        )

        # Get the current data from Firestore
        doc = doc_ref.get()

        # Check if the document exists
        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                # If the document exists, update the "missing_pdfs" field with new data
                existing_missing_pdfs = doc_data.get(FIRESTORE_REFERENCE_NAME, [])

                for missing_pdf in existing_missing_pdfs:
                    if missing_pdf.get("pdf_name") in selected_pdfs:
                        # Check if the category key already exists
                        if "category" in missing_pdf:
                            # If pdf_name matches, update the category
                            missing_pdf["category"] = selected_category
                        else:
                            # If pdf_name matches and category doesn't exist, add the category
                            missing_pdf["category"] = selected_category

                # Update the Firestore document with the merged data
                doc_ref.set(doc_data)
                return jsonify({"status": 200})
            else:
                return jsonify({"status": 400})

        else:
            doc_ref.set({FIRESTORE_REFERENCE_NAME: []})
            return jsonify({"status": 200})

    except Exception as upload_error:
        return jsonify({"status": 400})


def delete_multiple_pdfs_from_firestore(selected_pdfs):
    try:
        doc_ref = db.collection(FIRESTORE_COLLECTION_NAME).document(
            FIRESTORE_DOCUMENT_NAME
        )

        doc = doc_ref.get()

        if doc.exists:
            doc_data = doc.to_dict()

            if doc_data:
                existing_missing_pdfs = doc_data.get(FIRESTORE_REFERENCE_NAME, [])

                updated_missing_pdfs = [
                    missing_pdf
                    for missing_pdf in existing_missing_pdfs
                    if missing_pdf.get("pdf_name") not in selected_pdfs
                ]

                doc_ref.set({FIRESTORE_REFERENCE_NAME: updated_missing_pdfs})
                return jsonify({"status": 200})
            else:
                return jsonify({"status": 400})

        else:
            doc_ref.set({FIRESTORE_REFERENCE_NAME: []})
            return jsonify({"status": 200})

    except Exception as delete_error:
        return jsonify({"status": 400})
