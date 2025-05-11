import os
import io
import pandas as pd
import numpy as np
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image

# reference = "/nethome/chuang475/LMUData/a-okvqa.tsv"
# reference = pd.read_csv(reference, sep="\t", header=0)
# print(reference.keys())
# print(reference.iloc[10])

# df = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/train-00000-of-00002-c1d24de3bacb5e0c.parquet")
# df_1 = pd.read_parquet("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/a-okvqa/train-00001-of-00002-6b4f3abe2dc385d0.parquet")
# df = pd.concat([df, df_1], axis=0).reset_index(drop=True)
# /coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00000-of-00007.parquet

def convert_m3cot(split='train', 
                  input_dir=["/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/validation-00000-of-00001.parquet"], 
                  output_dir=["/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_VAL.tsv"]):
    # read parquet files and concatenate
    df = pd.concat([pd.read_parquet(file) for file in input_dir], axis=0).reset_index(drop=True)
    print(len(df))
    # print(df.index)
    # print(df.keys())
    # print(df.iloc[10])

    # ['id', 'category', 'image_id', 'question', 'choices', 'context',
    #  'answer', 'rationale', 'split', 'image', 'domain', 'topic']

    # change the column ['image', 'question_id', 'question', 'choices', 'correct_choice_idx', 'direct_answers', 'difficult_direct_answer', 'rationales']
    # to ['index', 'question', 'hint', 'A', 'B', 'C', 'D', 'answer', 'category', 'image', 'source', 'comment', 'split'] + 'solution'
    # the choices are a list of <= 5 items

    # Add the new columns for A, B, C, D, E (choices split into separate columns)
    choices_split = pd.DataFrame(df['choices'].to_list(), columns=['A', 'B', 'C', 'D', 'E'])
    print(choices_split.head())
    df = pd.concat([df, choices_split], axis=1)

    # Add 'split' column if needed (you can fill it with a default value or logic)
    df['split'] = split  # For example, default split can be 'train'

    # Add 'index' as an additional column (this will just create a new column with the index values)
    df['index'] = df.id

    # {'bytes': b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHD... }
    df['image'] = df['image'].apply(lambda x: x['bytes'] if x is not None else x)

    # map 0,1,2,3 to A,B,C,D
    # df['answer'] = df['correct_choice_idx'].map({0: 'A', 1: 'B', 2: 'C', 3: 'D'})
    df['solution'] = df['rationale']

    # delete choices
    df = df.drop(columns=['choices', 'rationale', 'id'])

    # hint: null, category: ALL, source: A-OKVQA, comment: null
    # df['hint'] = None
    # df['source'] = 'A-OKVQA'
    # df['comment'] = None
    # df['category'] = 'ALL'

    # Print the updated columns
    print("Updated columns:", df.columns)

    def decode_base64_to_image(base64_string, target_size=-1):
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        if image.mode in ('RGBA', 'P', 'LA'):
            image = image.convert('RGB')
        if target_size > 0:
            image.thumbnail((target_size, target_size))
        return image

    def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
        image = decode_base64_to_image(base64_string, target_size=target_size)
        base_dir = osp.dirname(image_path)
        if not osp.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        image.save(image_path)

    df['image'] = df['image'].apply(lambda x: base64.b64encode(x).decode('utf-8') if x is not None else x)
    # filter out image == None
    df = df[df['image'].notna()]

    # Show a sample of the DataFrame
    print(df.head())
    print(df.iloc[10])

    # decode_base64_to_image_file(base64.b64encode(df.image[10]).decode('utf-8'), "/coc/testnvme/chuang475/projects/VLMEvalKit/try.jpg")
    df.to_csv(output_dir[0], sep="\t", index=False, header=True) # , quoting=1, quotechar='"'

if __name__ == "__main__":
    # split = ['train', 'val', 'test']
    # input_dir_train = ["/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00000-of-00007.parquet",
    #                    "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00001-of-00007.parquet",
    #                    "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00002-of-00007.parquet",
    #                    "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00003-of-00007.parquet",
    #                    "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00004-of-00007.parquet",
    #                    "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00005-of-00007.parquet",
    #                    "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/train-00006-of-00007.parquet"]
    # input_dir_val = ["/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/validation-00000-of-00001.parquet"]
    # input_dir_test = ["/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/test-00000-of-00002.parquet",
    #                   "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/M3CoT/data/test-00001-of-00002.parquet"]
    # output_dir_train = ["/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TRAIN.tsv"]
    # output_dir_val = ["/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_VAL.tsv"]
    # output_dir_test = ["/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TEST.tsv"]
    # for split, input_dir, output_dir in zip(split, [input_dir_train, input_dir_val, input_dir_test], [output_dir_train, output_dir_val, output_dir_test]):
    #     convert_m3cot(split=split, input_dir=input_dir, output_dir=output_dir)

    from datasets import load_dataset

    # Load the ScienceQA dataset
    dataset = load_dataset("LightChen2333/M3CoT")

    # Access the training split
    train_data = dataset["train"]

    print(train_data[1])

    # find id == physics-959, save the image of PIL to a file
    for item in train_data:
        if item['id'] == 'physics-959':
            print(item)
            image = item['image']
            image.save("/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/physics-959.jpg")
            break

    # load the tsv file
    df = pd.read_csv("/coc/pskynet4/chuang475/datasets/LMUData/M3CoT_TRAIN.tsv", sep="\t", header=0)
    for i in range(len(df)):
        if df.iloc[i]['index'] == 'physics-959':
            print(df.iloc[i])
            image = df.iloc[i]['image']

            def decode_base64_to_image(base64_string, target_size=-1):
                image_data = base64.b64decode(base64_string)
                image = Image.open(io.BytesIO(image_data))
                if image.mode in ('RGBA', 'P', 'LA'):
                    image = image.convert('RGB')
                if target_size > 0:
                    image.thumbnail((target_size, target_size))
                return image
            
            def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
                image = decode_base64_to_image(base64_string, target_size=target_size)
                base_dir = osp.dirname(image_path)
                if not osp.exists(base_dir):
                    os.makedirs(base_dir, exist_ok=True)
                image.save(image_path)
            
            decode_base64_to_image_file(image, "/coc/testnvme/chuang475/projects/VLMEvalKit/convert_tsv/try.jpg")
            break
    