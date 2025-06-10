"""Ce module lie ensemble les modules de génération de questions
et de génération de réponses incorrectes
"""
from question_extraction import QuestionExtractor
from incorrect_answer_generation import IncorrectAnswerGenerator
import re
from nltk import sent_tokenize


class QuestionGeneration:
    """Cette classe contient la méthode pour générer des questions"""

    def __init__(self, num_questions, num_options):
        self.num_questions = num_questions
        self.num_options = num_options
        self.question_extractor = QuestionExtractor(num_questions)

    def clean_text(self, text):
        '''Nettoie le texte en préservant la ponctuation essentielle'''
        text = text.replace('\n', ' ')  # supprimer les retours à la ligne
        sentences = sent_tokenize(text)
        cleaned_text = ""

        for sentence in sentences:
            # AMÉLIORATION : préserver la ponctuation importante
            cleaned_sentence = re.sub(r'[^\s\w\.\?\!\,\;\:]', '', sentence)
            # substituer les espaces multiples par un seul espace
            cleaned_sentence = re.sub(' +', ' ', cleaned_sentence)
            cleaned_text += cleaned_sentence

            # CORRECTION du bug : impossible d'assigner à un caractère de string
            if cleaned_text.endswith(' '):
                cleaned_text = cleaned_text[:-1] + '.'
            else:
                cleaned_text += '.'

            cleaned_text += ' '  # ajouter un espace à la fin

        return cleaned_text.strip()  # supprimer les espaces en début/fin

    def generate_questions_dict(self, document):
        """Génère un dictionnaire de questions à partir d'un document"""
        try:
            # Validation du document d'entrée
            if not document or not document.strip():
                print("Erreur : Document vide ou invalide")
                return {}

            document = self.clean_text(document)
            self.questions_dict = self.question_extractor.get_questions_dict(document)

            if not self.questions_dict:
                print("Aucune question générée à partir du document")
                return {}

            self.incorrect_answer_generator = IncorrectAnswerGenerator(document)

            for i in range(1, self.num_questions + 1):
                if i not in self.questions_dict:
                    continue

                try:
                    self.questions_dict[i]["options"] = \
                        self.incorrect_answer_generator.get_all_options_dict(
                            self.questions_dict[i]["answer"],
                            self.num_options
                        )
                except Exception as e:
                    print(f"Erreur lors de la génération des options pour la question {i}: {e}")
                    # Créer des options par défaut en cas d'erreur
                    self.questions_dict[i]["options"] = {
                        1: self.questions_dict[i]["answer"],
                        2: "Option 2",
                        3: "Option 3",
                        4: "Option 4"
                    }

            return self.questions_dict

        except Exception as e:
            print(f"Erreur lors de la génération des questions : {e}")
            return {}