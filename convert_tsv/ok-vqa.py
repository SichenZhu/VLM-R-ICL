# "index"	"question"	"answer"	"image_path"
# --ok_vqa_train_image_dir_path "/srv/datasets/coco/train2014" \
# --ok_vqa_train_annotations_json_path "/srv/datasets/ok-vqa_dataset/mscoco_train2014_annotations.json" \
# --ok_vqa_train_questions_json_path "/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_train2014_questions.json" \
# --ok_vqa_test_image_dir_path "/srv/datasets/coco/val2014" \
# --ok_vqa_test_annotations_json_path "/srv/datasets/ok-vqa_dataset/mscoco_val2014_annotations.json" \
# --ok_vqa_test_questions_json_path "/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_val2014_questions.json" \

# convert the json to tsv
from vlmeval.smp import *

# questions = load("/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_train2014_questions.json")['questions']
# annotations = load("/srv/datasets/ok-vqa_dataset/mscoco_train2014_annotations.json")['annotations']

# index_list = []
# question_list = []
# answer_list = []
# image_path_list = []

# for question, annotation in zip(questions, annotations):
#     assert question['question_id'] == annotation['question_id']
#     index_list.append(annotation['question_id'])
#     question_list.append(question['question'])
#     answer_list.append(str([a['answer'] for a in annotation['answers']]))
#     image_path_list.append(f"/srv/datasets/coco/train2014/COCO_train2014_{str(annotation['image_id']).zfill(12)}.jpg")

# df = pd.DataFrame({
#     'index': index_list,
#     'question': question_list,
#     'answer': answer_list,
#     'image_path': image_path_list
# })
# print(df.head())
# dump(df, "/coc/pskynet4/chuang475/datasets/LMUData/OK-VQA_TRAIN.tsv")

def convert_ok_vqa(split, input_dir, output_dir):
    # Load the JSON files
    questions = load(input_dir[0])['questions']
    annotations = load(input_dir[1])['annotations']

    # Initialize lists to store data
    index_list = []
    question_list = []
    answer_list = []
    image_path_list = []

    # Iterate through the questions and annotations
    for question, annotation in zip(questions, annotations):
        assert question['question_id'] == annotation['question_id']
        index_list.append(annotation['question_id'])
        question_list.append(question['question'])
        answer_list.append(str([a['answer'] for a in annotation['answers']]))
        image_path_list.append(f"{input_dir[2]}/COCO_{split}2014_{str(annotation['image_id']).zfill(12)}.jpg")

    # Create a DataFrame
    df = pd.DataFrame({
        'index': index_list,
        'question': question_list,
        'answer': answer_list,
        'image_path': image_path_list
    })

    # Save the DataFrame to TSV
    dump(df, output_dir[0])
    print(f"Saved {split} data to {output_dir[0]}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir_train = [
        "/data1/szhu337/VLM_proj/VQA-ICL/data/ok-vqa_dataset/OpenEnded_mscoco_train2014_questions.json",
        "/data1/szhu337/VLM_proj/VQA-ICL/data/ok-vqa_dataset/mscoco_train2014_annotations.json",
        "/data1/szhu337/VLM_proj/VQA-ICL/data/coco/train2014"
    ]
    input_dir_test = [
        "/data1/szhu337/VLM_proj/VQA-ICL/data/ok-vqa_dataset/OpenEnded_mscoco_val2014_questions.json",
        "/data1/szhu337/VLM_proj/VQA-ICL/data/ok-vqa_dataset/mscoco_val2014_annotations.json",
        "/data1/szhu337/VLM_proj/VQA-ICL/data/coco/val2014"
    ]
    output_dir_train = ["/data1/szhu337/VLM_proj/VQA-ICL/data/VLM-R-ICL_data/LMUData/OK-VQA_TRAIN.tsv"]
    output_dir_test = ["/data1/szhu337/VLM_proj/VQA-ICL/data/VLM-R-ICL_data/LMUData/OK-VQA_VAL.tsv"]

    # Convert OK-VQA data
    convert_ok_vqa("train", input_dir_train, output_dir_train)
    convert_ok_vqa("val", input_dir_test, output_dir_test)