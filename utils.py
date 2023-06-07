import google.generativeai as palm
from google.cloud import vision
import json


def detect_text(content) -> list:
    """
    Detects text in the file.
    :param content: image as bytes
    :return: list of text annotations
    """

    def assemble_word(word):
        assembled_word=""
        for symbol in word.symbols:
            assembled_word+=symbol.text
        return assembled_word
    
    
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    document  = response.full_text_annotation
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    
    blocks_texts = []

    for page in document.pages:
        for block in page.blocks:
            block_text = []
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    assembled_word = assemble_word(word)
                    block_text.append(assembled_word)
            
            blocks_texts.append(' '.join(block_text))

    return blocks_texts


def instruct_agent(text:str, instruction:str=None, nb_words:int=500) -> str:
    """
    Use PaLM for answering a question
    :param text: context
    :param instruction: instruction to follow
    :param nb_words: maximum number of words for summarization
    :return: return answer as string
    """

    short_context = ' '.join(text.split()[:3500])

    if len(instruction) > 0:
        prompt = f"""
        You are acting as a scientific expert in a research project. 
        \n\n
        Follow the instruction using the information in the following context. 
        If you have to answer a question but no answer is found, say 'I don't know'.
        ##
        context: {short_context} \n
        instruction: {instruction}
        \n\n
        """

        models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
        model = models[0].name

        completion = palm.generate_text(
            model=model,
            prompt=prompt,
            temperature=0,
            max_output_tokens=800,
        )

        response = completion.result

    else:
        response = "No instruction given..."

    return(response)


def get_entities(text:str, vocabulary:str) -> str:
    """
    Use PaLM for extracting entities from a text
    :param text: context
    :param vocabulary: controlled vocabulary as csv string
    :return: return entities as csv string
    """

    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    # split text into sentences the easy way
    sentences = text.split('.')

    # place holder for extracted entities
    extracted_entities = dict()

    # do the extraction of entities for each sentence
    for sentence in sentences:

        prompt = f"""
        Instructions to perform the following tasks:
        1- To determine if a word in the context is an extracted entity, find the very similar entity by meaning in the vocabulary.
        2- Keep extracted entities as short as possible.
        3- For each extracted entity, use the meaning of the words to determine the closest word in the vocabulary.

        Following the previous instructions, perform the following task:
        A- Extract entity, its corresponding category from the context given below and the reference of the entity in the vocabulary.
        B- Format the output as a json using 'entity' as the key and with 'category' and 'reference' as values.
        C- Let's think step by step.
        ##
        context: {sentence}
        vocabulary: {vocabulary}
        \n\n
        """

        if len(vocabulary) > 0 and len(sentence) > 0:
            
            completion = palm.generate_text(
                model=model,
                prompt=prompt,
                temperature=0,
                max_output_tokens=800,
            )

            response = completion.result
            response = response.replace('```json\n', '').replace('```', '')
            response = json.loads(response)

            # check the type of the value of the key 'entity'
            first_key = list(response.keys())[0]
            if isinstance(response[first_key], dict):
                extracted_entities.update(response)

    if len(extracted_entities) == 0:
        extracted_entities = "Nothing extracted ..."

    return(extracted_entities)