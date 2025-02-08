# here is myllmservice.py


import logging


# logger = logging.getLogger(__name__)
import asyncio
from llmservice.base_service import BaseLLMService
from llmservice.generation_engine import GenerationRequest, GenerationResult
from typing import Optional, Union


# add default model param to init.

class MyLLMService(BaseLLMService):
    def __init__(self, logger=None, max_concurrent_requests=5):
        super().__init__(
            # logger=logger,
            logger=logging.getLogger(__name__),
            default_model_name="gpt-4o-mini",
            yaml_file_path='categorizer/prompts.yaml',
            max_rpm=60,
            max_concurrent_requests=max_concurrent_requests,
        )
        # No need for a semaphore here, it's handled in BaseLLMService
    
    def filter_simple(self, semantic_filter, text_to_be_filtered: str,) -> GenerationResult:
        data_for_placeholders = {
            'semantic_filter_text': semantic_filter_text,
        }
        
        p= "prompt"
        
        order = ["semantic_filter_text",  "semantic_filtering_task_desc", "output_formatting"]

        unformatted_prompt = self.generation_engine.craft_prompt(data_for_placeholders, order)


        unformatted_prompt=f"""
            Here is semantic text prompt for filtering below text {semantic_filter}

            here is text to be filtered {text_to_be_filtered}

            i want you to use given semantic text prompt to filter given text 

            give the output in pure answer format wihtout any introduction

        """


        
        pipeline_config = [
            {
                'type': 'SemanticIsolation',
                'params': {
                    'semantic_element_for_extraction': 'pure boolean answer True or False'
                }
            }
            # {
            #     'type': 'ConvertToDict',
            #     'params': {}
            # },
            # {
            #     'type': 'ExtractValue',
            #     'params': {'key': 'answer'}  # Extract the 'answer' key from the dictionary
            # }
        ]

        generation_request = GenerationRequest(
            data_for_placeholders=data_for_placeholders,
            unformatted_prompt=unformatted_prompt,
            model="gpt-4o-mini",
            output_type="str",
            # use_string2dict=False,
            operation_name="categorize_simple",
            pipeline_config= pipeline_config,
            # request_id=request_id
        )

        # self.logger.info("now will enter execute_generation...(myllmservice)")

        # for i in range(3):

        generation_result = self.execute_generation(generation_request)

        # self.logger.info("came back to llmservice...(myllmservice)")
            # if generation_result.

        return generation_result


   


def main():
    """
    Main function to test the categorize_simple method of MyLLMService.
    """
    # Initialize the service
    my_llm_service = MyLLMService()

    # Sample data for testing
    sample_search_title_and_metadescription = "BAV99 Fast Switching Speed. • Surface-Mount Package Ideally Suited for Automated Insertion. • For General-Purpose Switching Applications."
   
   
    try:
        # Perform categorization
        result = my_llm_service.categorize_simple(
            semantic_filter_text=sample_search_title_and_metadescription,
    
        )

        # Print the result
        print("Generation Result:", result)
        if result.success:
            print("Filtered Content:", result.content)
        else:
            print("Error:", result.error_message)
    except Exception as e:
        print(f"An exception occurred: {e}")


if __name__ == "__main__":
    main()
