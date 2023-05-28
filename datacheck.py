import boto3
import pandas as pd
from sagemaker import get_execution_role

role = get_execution_role()

# Specify the S3 bucket
bucket_name = 'llminfo'

# Instantiate a s3 client
s3 = boto3.client('s3', region_name='us-west-2')

# List all the objects in the bucket
objects_in_bucket = s3.list_objects(Bucket=bucket_name)

# Suppose you have a CSV file you want to read into a pandas dataframe
file_to_read = 'FaCov_dataset (3).csv' # replace with your file name

# Check if the file exists in the bucket
if any(obj['Key'] == file_to_read for obj in objects_in_bucket['Contents']):
    
    # Get the file object
    file_obj = s3.get_object(Bucket=bucket_name, Key=file_to_read)
    
    # Read the file object into a pandas dataframe
    data = pd.read_csv(file_obj['Body'])
    # print(df.head())
    # print(df.columns)
    # print(df['content'][0])
    # print(df['raw_label'][0])
    # print(df.raw_label.unique())
    selected_values = ['mostly true', 'half true', 'true', 'partly true', 'Partly True']
    sub_df = data[data['raw_label'].isin(selected_values)]
    print(sub_df.head())
    length = len(sub_df)
    print("Length of the subdataframe:", length)
    print(sub_df.columns)
    print(sub_df['content'].tolist()[0])
    print('********')
    content_string = '\n'.join(sub_df['content'].astype(str))


print(content_string)
