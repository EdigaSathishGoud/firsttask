from azure.cognitiveservices.vision.contentmoderator import ContentModeratorClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.contentmoderator.models import Screen
from pprint import pprint

SUBSCRIPTION_KEY = "109da6925fa2467784a7ec260619d963"
ENDPOINT = "https://guardrialscm.cognitiveservices.azure.com/"
REGION ="eastus"

client = ContentModeratorClient(endpoint=ENDPOINT, credentials=CognitiveServicesCredentials(SUBSCRIPTION_KEY))

TEXT_MODERATION_FILE=r"C:\Users\saediga\Downloads\content_moderator_text_moderation.txt"

with open(TEXT_MODERATION_FILE, "rb") as text_file:
    screen = client.text_moderation.screen_text(language="eng", text_content_type="text/plain", text_content=text_file, autocorrect=True, pii=True, classify=True)
    assert isinstance(screen, Screen)
    # Format and print
    text_mod_results = list(screen.as_dict().items())
    pprint(text_mod_results)
    print()