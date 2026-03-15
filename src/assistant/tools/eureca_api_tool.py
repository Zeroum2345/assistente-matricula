from langchain.tools import tool
import requests


@tool
def get_course_prerequisites(course_name: str) -> str:
    """
    Retorna os pré-requisitos de uma disciplina.
    """

    url = f"https://api.university.edu/courses/{course_name}"

    response = requests.get(url)

    data = response.json()

    return str(data["prerequisites"])