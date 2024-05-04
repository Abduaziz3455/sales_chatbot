from google.cloud import translate

client = translate.TranslationServiceClient()
project_id: str = "sales-chatbot-translate"
model_id: str = "general/nmt"
location = "us-central1"
parent = f"projects/{project_id}/locations/{location}"
model_path = f"{parent}/models/{model_id}"


def translate_text_with_model(
        text: str = "Salom",
        source_language_code: str = "uz",
        target_language_code: str = "en",
):
    # Supported language codes: https://cloud.google.com/translate/docs/languages
    response = client.translate_text(
        request={
            "contents": [text],
            "target_language_code": target_language_code,
            "model": model_path,
            "source_language_code": source_language_code,
            "parent": parent,
            "mime_type": "text/plain",  # mime types: text/plain, text/html
        }
    )
    # Display the translation for each input text provided
    translations = " ".join([x.translated_text for x in response.translations])
    return translations


# while True:
#     query = input("Enter your query: ")
#     print(translate_text_with_model(query))
