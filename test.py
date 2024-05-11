from langsmith import Client
from dotenv import load_dotenv
load_dotenv()

client = Client()
runs = client.list_runs(
          project_name="sales_chatbot",
          filter="and(eq(metadata_key, 'session_id'), eq(metadata_value, 1556455886))",
          is_root=True,
          run_type='chain'
        )

run_id = list(runs)[0].id
client.create_feedback(run_id, key='CORRECTNESS', score=1)
