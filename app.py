import datetime
import uuid

import dotenv
import os

import qdrant_client
from llama_index.postprocessor import FixedRecencyPostprocessor
from llama_index.schema import TextNode
from slack_bolt import App
from flask import Flask, request, jsonify
from slack_bolt.adapter.flask import SlackRequestHandler
from llama_index import VectorStoreIndex, StorageContext, PromptTemplate, ServiceContext, set_global_handler
from llama_index.vector_stores.qdrant import QdrantVectorStore

set_global_handler("simple")

dotenv.load_dotenv()

flask_app = Flask(__name__)

app = App(
    token=os.environ.get("SLACK_BOT_TOKEN"),
    signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
)

handler = SlackRequestHandler(app)


# setup qdrant local vdb as storage
client = qdrant_client.QdrantClient(
    path="./qdrant_data"
)
vector_store = QdrantVectorStore(client=client, collection_name="slack_messages")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex([], storage_context=storage_context)

# join #bot-testing channel
channel_id = None
channel_list = app.client.conversations_list().data
for channel in channel_list.get('channels'):
    if channel.get('name') == 'bot-testing':
        channel_id = channel.get('id')

app.client.conversations_join(channel=channel_id)

# bot's userID
auth_response = app.client.auth_test()
bot_user_id = auth_response["user_id"]


def store_message(message):
    dt_object = datetime.datetime.fromtimestamp(float(message.get('ts')))
    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M:%S')

    text = message.get('text')

    node = TextNode(
        text=text,
        id_=str(uuid.uuid4()),
        metadata={
            "when": formatted_time
        }
    )
    index.insert_nodes([node])
    print(f"Stored message: {text}")


def answer_question(query, message, replies=None):
    template = (
        "Your context is a series of chat messages. Each one is tagged with 'who:' \n"
        "indicating who was speaking and 'when:' indicating when they said it, \n"
        "followed by a line break and then what they said. There can be up to 20 chat messages.\n"
        "The messages are sorted by recency, so the most recent one is first in the list.\n"
        "The most recent messages should take precedence over older ones.\n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "You are a helpful AI assistant who has been listening to everything everyone has been saying. \n"
        "Given the most relevant chat messages above, please answer this question: {query_str}\n"
    )
    qa_template = PromptTemplate(template)
    postprocessor = FixedRecencyPostprocessor(
        top_k=20,
        date_key="when",
        service_context=ServiceContext.from_defaults()
    )
    query_engine = index.as_query_engine(similarity_top_k=20, node_postprocessors=[postprocessor])
    query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": qa_template}
    )
    return query_engine.query(query)


@flask_app.route("/", methods=["POST"])
def slack_challenge():
    if request.json and "challenge" in request.json:
        print("Received challenge")
        return jsonify({"challenge": request.json["challenge"]})
    else:
        print("Got unknown request incoming")
        print(request.json)
    return handler.handle(request)


@app.message()
def reply(message, say):
    print("Reply message")
    print(message)
    for block in message.get('blocks', []):
        if block.get('type') == 'rich_text':
            for rich_text_section in block.get('elements'):
                for element in rich_text_section.get('elements'):
                    print(bot_user_id, element)
                    if element.get('type') == 'user' and element.get('user_id') == bot_user_id:
                        for element in rich_text_section.get("elements"):
                            if element.get('type') == 'text':
                                query = element.get('text')
                                print(f"Somebody asked the bot: {query}")
                                response = answer_question(query, message)
                                say(str(response))
                                return


if __name__ == "__main__":
    flask_app.run(port=3000)