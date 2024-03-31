from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import QAGenerationChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings.vertexai import VertexAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
import os 
import json
import time
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import csv
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from langchain_google_vertexai import VertexAI
import vertexai
from vertexai.language_models import TextGenerationModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# PORT = os.environ.get("PORT")  # Get PORT setting from environment. Usually 8000
# if not PORT:
#     PORT = 8080

templates = Jinja2Templates(directory="templates")

# os.environ["OPENAI_API_KEY"] = ""

# Initialize Vertex AI access.
vertexai.init(project="clouda2-418901", location="us-central1")
MODEL = "text-bison@001"
parameters = {
    "model_name": MODEL,            
    # "candidate_count": 1,   
    "max_output_tokens": 1024,
    "temperature": 0.3,       
    "top_p": 0.8,             
    "top_k": 40,              
}

ans_params = {
    "model_name": MODEL,            
    # "candidate_count": 1,  
    "max_output_tokens": 1024,
    "temperature": 0.1,       
    "top_p": 0.8,             
    "top_k": 40,  
}
llm_model = VertexAI(**parameters)
llm_ans_model = VertexAI(**ans_params)

model_results = ""

def count_pdf_pages(pdf_path):
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        print("Error:", e)
        return None

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    splitter_QA_gen = TokenTextSplitter(
        model_name = "gpt-3.5-turbo",
        chunk_size = 1000,
        chunk_overlap = 200
    )

    QA_gen = ''

    for page in data:
        QA_gen += page.page_content
        

    chunks_QA_gen = splitter_QA_gen.split_text(QA_gen)

    document_QA_gen = [Document(page_content=t) for t in chunks_QA_gen]

    return document_QA_gen

def llm_pipeline(file_path):

    document_QA_gen = file_processing(file_path)



    prompt_template = """
    You are an expert at creating questions from the content in documents.
    Your goal is to prepare a student for their exam and coding tests.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create multiple choice questions(MCQ) with 4 choices that will prepare the student for their tests.
    Make sure not to lose any important information.

    Create at least 5 questions. More questions are preferred.

    Generate the output in this format: "Question", "Choices", "Correct answer", "Explanation".
    

    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, 
                                      input_variables=["text"], 
                                      )

    refine_template = ("""
    You are an expert at creating practice questions based on content in documents.
    Your goal is to help a student prepare for a coding test.
    We have received some practice questions to a certain extent: {existing_answer}.
    
    The practice questions follow this format: "Question", "Choices", "Correct answer", "Explanation".
                       
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
                       
    Create at least 5 questions. More questions are preferred.
                       
    Generate the output in this format: "Question", "Choices", "Correct answer", "Explanation".
    
    """
    )

    # Test
    # response_schemas = [
    # ResponseSchema(name="question", description="Question generated from provided input text data."),
    # ResponseSchema(name="choices", description="Available options for a multiple-choice question in comma separated."),
    # ResponseSchema(name="answer", description="Correct answer for the asked question.")
    # ]

    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # output_parser

    # format_instructions = output_parser.get_format_instructions()
 
    # print(format_instructions)
    
    # Add to prompt template: partial_variables={"format_instructions": format_instructions}

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_model, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    Query_return = ques_gen_chain.run(document_QA_gen)
    Query_return_list = Query_return.split("\n")

    global model_results
    model_results = "\n".join(Query_return_list)
    print(model_results)
    
    # filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]
    ques_list = [element for element in Query_return_list if element.startswith('Question') or element.endswith('?')]
    choice_A_list = [element for element in Query_return_list if element.startswith('(A)')]
    choice_B_list = [element for element in Query_return_list if element.startswith('(B)')]
    choice_C_list = [element for element in Query_return_list if element.startswith('(C)')]
    choice_D_list = [element for element in Query_return_list if element.startswith('(D)')]
    correct_list = [element for element in Query_return_list if element.startswith('Correct')]
    Explanation_list = [element for element in Query_return_list if element.startswith('Explanation')]

    # return filtered_ques_list
    return ques_list, choice_A_list, choice_B_list, choice_C_list, choice_D_list, correct_list, Explanation_list

def get_csv (file_path):
    # answer_generation_chain, ques_list = llm_pipeline(file_path)
    # ques_list = llm_pipeline(file_path)
    ques_list, choice_A_list, choice_B_list, choice_C_list, choice_D_list, correct_list, Explanation_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "choice_A_list", "choice_B_list", 
                             "choice_C_list", "choice_D_list", 
                             "correct_list", "Explanation_list"])  # Writing the header row

        for i, question in enumerate(ques_list):
        #     print("Question: ", question)
        #     answer = answer_generation_chain.run(question)
        #     print("Answer: ", answer)
        #     print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            # csv_writer.writerow([question, answer])
            csv_writer.writerow([ques_list[i], choice_A_list[i], 
                                 choice_B_list[i], choice_C_list[i], 
                                 choice_D_list[i], correct_list[i], 
                                 Explanation_list[i]])
    return output_file

def file_processing1(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 10000,
        chunk_overlap = 200
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = TokenTextSplitter(
        model_name = 'gpt-3.5-turbo',
        chunk_size = 1000,
        chunk_overlap = 100
    )


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen

def llm_pipeline1(file_path):

    document_ques_gen, document_answer_gen = file_processing1(file_path)

    prompt_template = """
    Given the text below, generate 5 multiple choice questions(MCQ) with 4 choices.

    ------------
    {text}
    ------------

    IMPORTANT: Give me 5 questions!!!

    Generate the output in this format: "Question", "Choices", "Correct answer", "Explanation".

    """
    # Other prompts
    # IMPORTANT: Give me 5 questions!!!
    # Give the output with explanations in JSON format
    # Generate the output in this format: "Question", "Choices", "Correct answer", "Explanation".
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, 
                                      input_variables=["text"], 
                                      )
    
    prompt_formatted_str: str = prompt_template.format(
                                text= document_ques_gen,
                            )

    ques = llm_model.predict(text= prompt_formatted_str)

    # ques_gen_chain = load_summarize_chain(llm = llm_model, 
    #                                         chain_type = "refine", 
    #                                         verbose = True, 
    #                                         question_prompt=PROMPT_QUESTIONS, 
    #                                         )

    # ques = ques_gen_chain.run(document_ques_gen)

    embeddings = VertexAIEmbeddings()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    ques_list = ques.split("\n")
    global model_results
    model_results = "\n".join(ques_list)
    print(model_results)
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_ans_model, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list

def get_csv1(file_path):
    # answer_generation_chain, ques_list = llm_pipeline(file_path)
    # ques_list = llm_pipeline(file_path)
    answer_generation_chain, ques_list = llm_pipeline1(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])
    return output_file

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)
    # page_count = count_pdf_pages(pdf_filename)
    # if page_count > 5:
    #     return Response(jsonable_encoder(json.dumps({"msg": 'error'})))
    response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
    res = Response(response_data)
    return res


@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    # output_file = get_csv1(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file, "model_results": model_results}))
    res = Response(response_data)
    return res

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), reload=True)
