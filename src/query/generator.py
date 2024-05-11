"""Manages SQL generation using llm"""

import json
import re
import uuid
import boto3
import sqlglot

from embeding.embed import get_similar_documents
import re

from query.executor import schema_validation


def get_relavant_schema(prompt: str) -> str:
    """Get relavant schema for query

    Args:
        prompt (str): _description_

    Returns:
        str: _description_
    """
    documents = get_similar_documents(prompt=prompt, collection_name="table_schema")
    return [d.page_content for d in documents]


def get_valid_examples(prompt: str) -> str:
    """Get relavant examples

    Args:
        prompt (str): _description_

    Returns:
        str: _description_
    """
    documents = get_similar_documents(
        prompt=prompt, collection_name="query_valid_examples"
    )
    return [d.page_content for d in documents]


def get_invalid_examples(prompt: str) -> str:
    """Get relavant examples

    Args:
        prompt (str): _description_

    Returns:
        str: _description_
    """
    documents = get_similar_documents(
        prompt=prompt, collection_name="query_invalid_examples"
    )
    return [d.page_content for d in documents]


def get_query(prompt: str):
    """Get query"""
    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "prompt": prompt,
        "max_tokens_to_sample": 2048,
        "temperature": 0,
    }

    response = bedrock.invoke_model(
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json",
        modelId="anthropic.claude-v2",
    )
    return json.loads(response.get("body").read())["completion"]


def get_prompt_template() -> str:
    return """Human: 

    <role>You are an expert SQL Query Translator system. Your role is to accurately translate natural language queries into syntactically correct SQL statements based on the provided database schema information. You should handle a wide range of query complexities involving filters, joins, aggregations, etc. If the query cannot be properly translated given the context, you should indicate that gracefully.</role>

    <document>
        <database_schema>
        '{context}'
        </database_schema>
    </document>

    <example>
        <valid>
        '{examples}'
        </valid>
        <invalid>
        '{error_examples}'
        </invalid>
    </example>

    <instructions>
        1. Identify the main entities and tables involved based on the provided schema.
        2. Determine the relationships between tables and how to join them using the schema information.
        3. Analyze the filters, aggregations, and calculations required based on the natural language query.
        4. Construct the SELECT, FROM, JOIN, WHERE, GROUP BY, and ORDER BY clauses using only the information available in the schema and query.
        5. Ensure the query retrieves the requested information accurately without making any assumptions beyond the given context.
        6. Do not hallucinate or generate any information that is not explicitly provided in the schema or query.
        7. Return the SQL query translation as a JSON object with the keys "result" (containing the SQL query) and "confidence" (a float value between 0 and 1 indicating your confidence in the translation).
        8. Always check examples before giving results and try to be as close to examples shared
        9. Refer error examples and try to avoid quries similar to error examples
        9. When you generate the answer, first think how the output should be structured and add your answer in <thinking></thinking> tags. This is a space for you to write down relevant content and will not be shown to the user. 
        10.Once you are done thinking, answer the question. Put your answer inside <answer></answer> XML tags.
    </instructions>

    <task> Given a natural language query and a database schema, translate the '{query}' into a valid SQL statement</task>

    Output result in the format provided below ```
    <format>
    {output}
    </format>

    \n\nAssistant:"
"""


def get_prompt_template_with_error() -> str:
    return """Human: 

    <role>You are an expert SQL Query Translator system. Your role is to accurately translate natural language queries into syntactically correct SQL statements based on the provided database schema information. You should handle a wide range of query complexities involving filters, joins, aggregations, etc. If the query cannot be properly translated given the context, you should indicate that gracefully.</role>

    Original SQL query given in previous promot which resulted in error:
    <original_sql_query>
        {original_sql_query}
    </original_sql_query>

    <error>
        Error occured because returned_at or completed_at was NULL
    </error>

    <document>
        <database_schema>
        '{context}'
        </database_schema>
    </document>

    <example>
        <valid>
        '{examples}'
        </valid>
        <invalid>
        '{error_examples}'
        </invalid>
    </example>

    <instructions>
        1. Identify the main entities and tables involved based on the provided schema.
        2. Determine the relationships between tables and how to join them using the schema information.
        3. Analyze the filters, aggregations, and calculations required based on the natural language query.
        4. Construct the SELECT, FROM, JOIN, WHERE, GROUP BY, and ORDER BY clauses using only the information available in the schema and query.
        5. Ensure the query retrieves the requested information accurately without making any assumptions beyond the given context.
        6. Do not hallucinate or generate any information that is not explicitly provided in the schema or query.
        7. Return the SQL query translation as a JSON object with the keys "result" (containing the SQL query) and "confidence" (a float value between 0 and 1 indicating your confidence in the translation).
        8. Always check examples before giving results and try to be as close to examples shared
        9. Refer error examples and try to avoid quries similar to error examples
        9. When you generate the answer, first think how the output should be structured and add your answer in <thinking></thinking> tags. This is a space for you to write down relevant content and will not be shown to the user. 
        10.Once you are done thinking, answer the question. Put your answer inside <answer></answer> XML tags.
    </instructions>

    <task> Given a natural language query and a database schema, translate the '{query}' into a valid SQL statement</task>

    Output result in the format provided below ```
    <format>
    {output}
    </format>

    \n\nAssistant:"
"""


def extract_answer_and_thinking(text):
    answer_pattern = r"<answer>\n(.*?)\n<\/answer>"
    thinking_pattern = r"<thinking>(.*?)<\/thinking>"

    answer_match = re.search(answer_pattern, text, re.DOTALL)
    thinking_match = re.search(thinking_pattern, text, re.DOTALL)

    answer_text = answer_match.group(1).strip() if answer_match else None
    thinking_text = thinking_match.group(1).strip() if thinking_match else None

    if answer_text:
        return json.loads(answer_text), thinking_text
    else:
        return None, thinking_text


def generate_sql(*, prompt: str) -> tuple[str, str]:
    """Generate query using LLM"""

    context = get_relavant_schema(prompt=prompt)
    valid_examples = get_valid_examples(prompt=prompt)
    invalid_examples = get_invalid_examples(prompt=prompt)

    output_template = '{"results":[{"result":"sql","confidence":0.0,"justification":"","example_used_for_query_generation":""}]"}'

    prompt = get_prompt_template().format(
        query=prompt,
        context=context,
        examples=valid_examples,
        error_examples=invalid_examples,
        output=output_template,
    )

    query = get_query(prompt=prompt)

    answer, thinking = extract_answer_and_thinking(text=query)
    generated_sql = answer["results"][0]["result"]

    result, details = schema_validation(
        query=generated_sql, report_id=str(uuid.uuid4())
    )

    # retry if result is not sucessful
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        match result:
            case "SUCCEEDED":
                return generated_sql, thinking
            case _:
                prompt = get_prompt_template_with_error().format(
                    query=prompt,
                    context=context,
                    examples=valid_examples,
                    error_examples=invalid_examples,
                    output=output_template,
                    original_sql_query=generated_sql,
                    error=details,
                )

                query = get_query(prompt=prompt)
                answer, thinking = extract_answer_and_thinking(text=query)
                generated_sql = answer["results"][0]["result"]

                result, details = schema_validation(
                    query=generated_sql, report_id=str(uuid.uuid4())
                )
        attempts += 1
