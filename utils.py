import google.generativeai as palm
from google.cloud import vision


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