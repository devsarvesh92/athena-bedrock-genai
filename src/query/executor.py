import time
import boto3
import uuid
import pandas as pd

athena = boto3.client(service_name="athena")
s3 = boto3.client(service_name="s3")
report_id: str = str(uuid.uuid4())


def execute_sql(query: str, report_id=str(uuid.uuid4())):
    bucket_name: str = "sandbox-data-query-service"
    result_bucket: str = f"s3://{bucket_name}/"
    report_location: str = f"{result_bucket}{report_id}"

    response = athena.start_query_execution(
        QueryString=query,
        ClientRequestToken=report_id,
        QueryExecutionContext={
            "Database": "sandbox_payments",
            "Catalog": "AwsDataCatalog",
        },
        ResultConfiguration={"OutputLocation": report_location},
        WorkGroup="primary",
    )

    time.sleep(5)

    obj = s3.get_object(
        Bucket=bucket_name, Key=f"{report_id}/{response['QueryExecutionId']}.csv"
    )
    data = pd.read_csv(obj["Body"])

    return data


def schema_validation(query: str, report_id:str):
    """
    Always add limit 1 to query for schema of athena based validation

    Args:
        query (str): _description_
    """
    bucket_name: str = "sandbox-data-query-service"
    result_bucket: str = f"s3://{bucket_name}/"
    report_location: str = f"{result_bucket}{report_id}"

    query = f"{query} limit 1"

    response = athena.start_query_execution(
        QueryString=query,
        ClientRequestToken=report_id,
        QueryExecutionContext={
            "Database": "sandbox_payments",
            "Catalog": "AwsDataCatalog",
        },
        ResultConfiguration={"OutputLocation": report_location},
        WorkGroup="primary",
    )

    terminating_state = {"SUCCEEDED", "FAILED", "CANCELLED"}
    current_state = "QUEUED"

    while current_state not in terminating_state:
        result = athena.get_query_execution(
            QueryExecutionId=response["QueryExecutionId"]
        )
        current_state = result["QueryExecution"]["Status"]["State"]

    return current_state, result["QueryExecution"]["Status"]
