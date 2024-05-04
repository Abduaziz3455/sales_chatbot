import json

with open('data.jsonl', 'r') as json_file:
    json_data = json.load(json_file)
user1 = json_data[:7]
user2 = json_data[7:14]
user3 = json_data[14:21]
user4 = json_data[21:28]
user5 = json_data[28:35]
user6 = json_data[35:42]
user7 = json_data[42:49]
user8 = json_data[49:56]
user9 = json_data[56:63]
user10 = json_data[63:72]

qa_list = []

for user_qa in [user1, user2, user3, user4, user5, user6, user7, user8, user9, user10]:
    qa_list_user = []
    for qa in user_qa:
        qa_list_user.append({"role": "user", "content": qa['savol']})
        qa_list_user.append({"role": "assistant", "content": qa['javob']})
    qa_list.append({"messages": qa_list_user})

with open('conversations.jsonl', 'w') as outfile:
    for qa in qa_list:
        json.dump(qa, outfile)
    outfile.close()
