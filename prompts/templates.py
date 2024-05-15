SUMMARY_SYSTEM_PROMPT = \
"""
You are an expert software engineer who has been asked to generate a summary of a class in a Java code file. The class contains attributes and methods.
"""

SUMMARY_TEMPLATE = \
"""
Given a class in a Java code file, generate a class summary to map a given use case requirements to the given java code. 
The summary should capture the purpose of this class such that given a use case requirement, it can be determined if this class is relevant.
The summary should be concise not more than 2-3 lines which contains Java Code Keywords present in the class that can be useful to map a usecase requirement to this java code.
"""


CLASS_TRACE_TEMPLATE = \
"""
What are the names of the classes that are related to the following use case?
{requirement}

Provide the answer in a list format and provide ONLY the list of class names as a JSON list.
[<"Class 1 Name">, <"Class 2 Name">, ... <"Class N Name">] where N can be up to 10.
"""